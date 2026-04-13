from __future__ import annotations

import html
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
from tinygrad.engine.realize import run_schedule
from tinygrad.uop.ops import Ops
from tinygrad.viz.serve import uop_to_json

from compiler import compile_top_module, show_hardware, synthesis_stats
from compiler.backend import HDLRenderer, _get_uops, uops_to_kernel_ir, _count_cycles_from_root
from compiler.top_module import simulate_top
from compiler.visualize import analyze_schedule


@dataclass
class PCIeModel:
    """Model for PCIe + AXI-DMA data transfer overhead.

    Represents an FPGA connected over PCIe with an AXI-DMA bridge.
    Transfer time per direction = latency_s + nbytes / bw_bytes_s.
    Default: Gen3 x4 (~3.2 GB/s practical) with 5 µs DMA latency.
    """
    gen: int = 3
    lanes: int = 4
    # FIXME: DMA latency is a fixed estimate (5 µs); real latency depends on
    # the specific AXI-DMA IP core, bridge configuration, and host-side driver.
    latency_s: float = 5e-6  # per-direction DMA setup + AXI bridge overhead

    # FIXME: Practical bandwidth assumes ~80% of theoretical link rate.
    # Real throughput varies with TLP payload size, MPS/MRRS settings,
    # and host chipset overhead.  Measure with a DMA loopback test.
    _BW_TABLE: ClassVar[dict] = {
        (1,  1): 0.20e9, (1,  4): 0.80e9, (1,  8): 1.60e9, (1, 16): 3.20e9,
        (2,  1): 0.40e9, (2,  4): 1.60e9, (2,  8): 3.20e9, (2, 16): 6.40e9,
        (3,  1): 0.80e9, (3,  4): 3.20e9, (3,  8): 6.40e9, (3, 16): 12.8e9,
        (4,  1): 1.60e9, (4,  4): 6.40e9, (4,  8): 12.8e9, (4, 16): 25.6e9,
        (5,  1): 3.20e9, (5,  4): 12.8e9, (5,  8): 25.6e9, (5, 16): 51.2e9,
    }

    @property
    def bw_bytes_s(self) -> float:
        return self._BW_TABLE.get((self.gen, self.lanes), 3.20e9)

    def xfer_s(self, nbytes: int) -> float:
        """One-direction transfer time: latency + bytes / bandwidth."""
        return self.latency_s + nbytes / self.bw_bytes_s


@dataclass
class BenchmarkArtifact:
    output_dir: str
    report_path: str
    tinygrad_device: str
    tinygrad_wall_s: float        # run_schedule() + .numpy() — compute + output copy
    tg2hdl_wall_s: float          # Amaranth simulator wall time (not hardware time)
    tg2hdl_cycles: int            # compute-only cycles (start → done)
    tg2hdl_total_cycles: int      # load + compute + readback cycles
    correctness: bool
    fpga_wall_s: float | None = None        # tg2hdl_total_cycles / fmax_hz; None if synthesis unavailable
    pcie_in_s: float | None = None          # PCIe host→FPGA transfer time
    pcie_out_s: float | None = None         # PCIe FPGA→host transfer time
    fpga_with_pcie_s: float | None = None   # fpga_wall_s + pcie_in_s + pcie_out_s


FPGA_FAMILY = "Lattice ECP5"
FPGA_DEVICE = "45k"
FPGA_PACKAGE = "CABGA381"


def _copy_viz_assets(out_dir: Path) -> None:
    import tinygrad

    src = Path(tinygrad.__file__).resolve().parent / "viz" / "assets"
    dst = out_dir / "assets"
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _buffer_to_numpy(buf) -> np.ndarray:
    return np.array(buf.numpy(), copy=True)


def _decode_sim_output(raw: np.ndarray, dtype) -> np.ndarray:
    name = getattr(dtype, "name", str(dtype))
    if name == "float":
        return raw.view(np.float32)
    if name == "half":
        return raw.view(np.uint16).view(np.float16)
    return raw


def _infer_input_data(schedule, pipeline_view, top) -> dict[tuple[int, int], np.ndarray]:
    copy_sources: dict[int, np.ndarray] = {}
    for si in schedule:
        if si.ast.op != Ops.COPY or len(si.bufs) < 2 or si.bufs[0] is None or si.bufs[1] is None:
            continue
        copy_sources[id(si.bufs[0])] = _buffer_to_numpy(si.bufs[1])

    compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
    input_data = {}
    for exec_k, buf_idx in top.ext_write_ports:
        orig_k = pipeline_view.execution_kernel_map[exec_k]
        buf = compute_items[orig_k].bufs[buf_idx]
        arr = copy_sources.get(id(buf))
        if arr is None and buf is not None and getattr(buf, "is_initialized", lambda: False)():
            arr = _buffer_to_numpy(buf)
        if arr is None:
            raise RuntimeError(f"Unable to infer external input data for kernel {orig_k} buffer {buf_idx}")
        input_data[(exec_k, buf_idx)] = arr
    return input_data


def _graph_payload(view) -> dict[str, dict]:
    return {
        "pipeline": view.graph_json(),
        "execution": view.execution_graph_json(),
    }


def _kernel_payload(schedule, pipeline_view, kernel_specs, synth_dir: Path) -> list[dict]:
    compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
    payload = []
    for exec_k, orig_k in enumerate(pipeline_view.execution_kernel_map):
        si = compute_items[orig_k]
        kv = pipeline_view.kernel_views[orig_k]
        kernel = kernel_specs[exec_k].kernel
        svg = None
        svg_path = show_hardware(kernel, str(synth_dir / f"kernel_{exec_k}"), fmt="svg")
        if svg_path is not None:
            svg = Path(svg_path).read_text()
        synth = synthesis_stats(kernel, device=FPGA_DEVICE, package=FPGA_PACKAGE)
        payload.append({
            "exec_index": exec_k,
            "source_index": orig_k,
            "metadata": [str(m.name if hasattr(m, "name") else m) for m in si.metadata],
            "tinygrad_graph": uop_to_json(si.ast),
            "kernel_ir": kv.kernel_ir.format(kv.kernel_ir),
            "synth_svg": svg,
            "synth_stats": synth,
        })
    return payload


def _top_payload(top, synth_dir: Path) -> dict:
    svg = None
    svg_path = show_hardware(top, str(synth_dir / "top_module"), fmt="svg")
    if svg_path is not None:
        svg = Path(svg_path).read_text()
    synth = synthesis_stats(top, device=FPGA_DEVICE, package=FPGA_PACKAGE)
    return {
        "name": "TopModule",
        "description": "Full assembled tg2hdl system including all compiled kernels, top-level control FSM, inter-kernel copies, and exposed I/O wiring.",
        "synth_svg": svg,
        "synth_stats": synth,
    }


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}s"


def _format_mhz(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} MHz"


def _pick_unit(*seconds_values: float | None) -> tuple[float, str]:
    """Return (multiplier, label) for the best shared time unit across all values."""
    vals = [v for v in seconds_values if v is not None and v > 0]
    ref = max(vals) if vals else 0.0
    if ref < 1e-6:
        return 1e9, "ns"
    if ref < 1e-3:
        return 1e6, "µs"
    if ref < 1.0:
        return 1e3, "ms"
    return 1.0, "s"


def _fmt_time(value: float | None, mult: float, unit: str) -> str:
    if value is None:
        return "n/a"
    return f"{value * mult:.3f} {unit}"


def _format_fpga_wall(cycles: int, fmax_mhz: float | None) -> str:
    if fmax_mhz is None or fmax_mhz <= 0:
        return "n/a"
    wall_s = cycles / (fmax_mhz * 1e6)
    mult, unit = _pick_unit(wall_s)
    return _fmt_time(wall_s, mult, unit)


def _bar_chart(title: str, rows: list, domain: float, mult: float, unit: str,
               W: int = 620, LEFT: int = 110, RIGHT: int = 140,
               ROW_H: int = 22, GAP: int = 8, TOP: int = 18, BOT: int = 28) -> str:
    """Render one horizontal stacked-bar chart as an inline SVG."""
    bar_w = W - LEFT - RIGHT
    H = TOP + len(rows) * ROW_H + (len(rows) - 1) * GAP + BOT

    def scale(v):
        return bar_w * v / domain if domain > 0 else 0

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'style="font-family:monospace;font-size:11px;display:block;margin:4px 0 8px 0;">']

    # title
    out.append(f'<text x="{LEFT}" y="13" font-weight="bold" fill="#111">{title}</text>')

    for ri, (label, segs) in enumerate(rows):
        y = TOP + ri * (ROW_H + GAP)
        cx = 0.0
        total = sum(v for _, v, _ in segs)

        out.append(f'<text x="{LEFT - 4}" y="{y + ROW_H - 6}" '
                   f'text-anchor="end" fill="#111">{label}</text>')

        for seg_label, val, color in segs:
            if val <= 0:
                continue
            px = scale(val)
            tooltip = f"{seg_label}: {val * mult:.4g} {unit}"
            out.append(f'<rect x="{LEFT + cx:.1f}" y="{y}" width="{max(px, 1):.1f}" '
                       f'height="{ROW_H}" fill="{color}"><title>{tooltip}</title></rect>')
            if px > 36:
                out.append(f'<text x="{LEFT + cx + px/2:.1f}" y="{y + ROW_H - 7}" '
                           f'text-anchor="middle" fill="#fff" font-size="10">{seg_label}</text>')
            cx += px

        total_str = f"{total * mult:.4g} {unit}"
        out.append(f'<text x="{LEFT + cx + 4:.1f}" y="{y + ROW_H - 6}" fill="#111">{total_str}</text>')

    # axis
    ay = H - BOT
    out.append(f'<line x1="{LEFT}" y1="{ay}" x2="{LEFT + bar_w}" y2="{ay}" stroke="#888" stroke-width="1"/>')
    for i in range(6):
        tx = LEFT + bar_w * i // 5
        tv = domain * i / 5 * mult
        out.append(f'<line x1="{tx}" y1="{ay}" x2="{tx}" y2="{ay + 4}" stroke="#888" stroke-width="1"/>')
        out.append(f'<text x="{tx}" y="{ay + 14}" text-anchor="middle" fill="#555">{tv:.3g}</text>')
    out.append(f'<text x="{LEFT + bar_w // 2}" y="{H - 2}" text-anchor="middle" fill="#555">{unit}</text>')

    out.append("</svg>")
    return "\n".join(out)


def _timing_svg(timing: dict) -> str:
    """Three charts: comparison (shared scale), CPU breakdown, FPGA+PCIe breakdown."""
    cpu_segs = [
        ("compute",  timing["cpu_compute_s"],  "#2a7db8"),
        ("readback", timing["cpu_readback_s"],  "#6ab0e0"),
    ]
    if timing["fpga_available"]:
        fpga_segs = [
            ("PCIe in",  timing["pcie_in_s"],      "#5a8fc4"),
            ("load",     timing["fpga_load_s"],     "#1f7a43"),
            ("compute",  timing["fpga_compute_s"],  "#d88412"),
            ("readback", timing["fpga_readback_s"], "#a05cb0"),
            ("PCIe out", timing["pcie_out_s"],      "#5a8fc4"),
        ]
        cpu_total  = timing["cpu_wall_s"]
        fpga_total = timing["fpga_with_pcie_s"]
        mult_cmp, unit_cmp   = _pick_unit(cpu_total, fpga_total)
        mult_cpu, unit_cpu   = _pick_unit(cpu_total)
        mult_fpga, unit_fpga = _pick_unit(fpga_total)

        cmp_chart  = _bar_chart(
            "Comparison (shared scale)",
            [("CPU",       [("total", cpu_total,  "#2a7db8")]),
             ("FPGA+PCIe", [("total", fpga_total, "#d88412")])],
            max(cpu_total, fpga_total), mult_cmp, unit_cmp,
        )
        cpu_chart  = _bar_chart(
            "CPU breakdown",
            [("CPU", cpu_segs)],
            cpu_total, mult_cpu, unit_cpu,
        )
        fpga_chart = _bar_chart(
            "FPGA+PCIe breakdown",
            [("FPGA+PCIe", fpga_segs)],
            fpga_total, mult_fpga, unit_fpga,
        )
        return cmp_chart + cpu_chart + fpga_chart

    else:
        # No fmax: comparison uses CPU time only; FPGA shows raw cycles (PCIe needs fmax)
        mult_cpu, unit_cpu = _pick_unit(timing["cpu_wall_s"])
        fpga_cycle_segs = [
            ("load",     timing["load_cycles"],     "#1f7a43"),
            ("compute",  timing["compute_cycles"],  "#d88412"),
            ("readback", timing["readback_cycles"], "#a05cb0"),
        ]
        cpu_chart = _bar_chart(
            "CPU breakdown",
            [("CPU", cpu_segs)],
            timing["cpu_wall_s"], mult_cpu, unit_cpu,
        )
        fpga_chart = _bar_chart(
            "FPGA breakdown (cycles, no Fmax — PCIe not modelled)",
            [("FPGA", fpga_cycle_segs)],
            timing["total_cycles"], 1.0, "cycles",
        )
        return cpu_chart + fpga_chart


# ---------------------------------------------------------------------------
# Flame graph payload + SVG renderer
# ---------------------------------------------------------------------------

def _flamegraph_payload(schedule, pipeline_view, top, kernel_specs,
                        cycle_counts: dict, tinygrad_wall: float,
                        cpu_compute_wall: float, cpu_readback_wall: float) -> dict:
    """Build flame graph data for both FPGA and CPU execution paths.

    Returns a dict with 'fpga' and 'cpu' keys, each containing a flat list
    of labeled spans with cycle/time values for SVG rendering.
    """
    compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
    renderer = HDLRenderer()

    # --- FPGA flame graph: per-kernel analytical cycle breakdown ---
    fpga_spans = []
    analytical_total = 0

    # Load phase (BRAM writes from simulation)
    fpga_spans.append({
        "label": "BRAM load",
        "cycles": cycle_counts["load"],
        "type": "load",
        "estimated": False,
    })

    # Per-kernel compute spans
    for exec_k, orig_k in enumerate(pipeline_view.execution_kernel_map):
        si = compute_items[orig_k]
        uops = _get_uops(si.ast, renderer)
        kernel_ir, _ = uops_to_kernel_ir(uops)
        # FIXME: Cycle count comes from an analytical model (_count_cycles_from_root)
        # that assumes perfect pipelining with no memory stalls or pipeline bubbles.
        # The real hardware FSM may differ due to startup/shutdown overhead,
        # wave grouping, and prologue/epilogue states.
        k_cycles = _count_cycles_from_root(kernel_ir.loop_tree)
        analytical_total += k_cycles
        meta = [str(m.name if hasattr(m, "name") else m) for m in si.metadata]
        name = f"K{exec_k}" + (f" ({', '.join(meta)})" if meta else "")
        fpga_spans.append({
            "label": name,
            "cycles": k_cycles,
            "type": "compute",
            "estimated": True,
        })

        # Inter-kernel copy groups that follow this kernel
        for gi, group in enumerate(top._copy_groups.get(exec_k, [])):
            src_buf = group[0][1]
            depth = top.buf_depths.get((exec_k, src_buf), 0)
            # FIXME: Copy cycle count equals buffer depth (one element per clock).
            # Real DMA copies may have additional setup latency or burst overhead
            # not captured here.
            dsts = [f"K{dk}.b{db}" for _, _, dk, db in group]
            fpga_spans.append({
                "label": f"copy K{exec_k}.b{src_buf}\u2192{','.join(dsts)}",
                "cycles": depth,
                "type": "copy",
                "estimated": True,
            })
            analytical_total += depth

    # FSM overhead residual
    fsm_overhead = max(0, cycle_counts["compute"] - analytical_total)
    if fsm_overhead > 0:
        # FIXME: FSM overhead is the residual between observed simulation cycles
        # and the sum of analytical per-kernel + copy cycles.  This lumps together
        # start/done handshake, state-transition latency, and any pipeline stalls.
        fpga_spans.append({
            "label": "FSM overhead",
            "cycles": fsm_overhead,
            "type": "overhead",
            "estimated": True,
        })

    # Readback phase (BRAM reads from simulation)
    fpga_spans.append({
        "label": "BRAM readback",
        "cycles": cycle_counts["readback"],
        "type": "readback",
        "estimated": False,
    })

    # --- CPU flame graph: per-schedule-item timing ---
    cpu_spans = []

    # Compute phase — distribute proportionally by buffer size
    # FIXME: Per-item timing for the CPU path is not available because
    # tinygrad's run_schedule() executes the entire schedule atomically.
    # We distribute the measured compute wall time proportionally based on
    # estimated compute cost (total buffer bytes) as a rough approximation.
    weights = []
    labels = []
    for idx, si in enumerate(schedule):
        bufs = getattr(si, "bufs", [])
        weight = sum(getattr(b, "nbytes", 0) for b in bufs if b is not None) or 1
        weights.append(weight)
        if si.ast.op == Ops.SINK:
            meta = [str(m.name if hasattr(m, "name") else m) for m in getattr(si, "metadata", ())]
            labels.append(f"K{idx}" + (f" ({', '.join(meta)})" if meta else ""))
        elif si.ast.op == Ops.COPY:
            labels.append(f"Copy {idx}")
        else:
            labels.append(f"Op {idx}")

    total_weight = sum(weights) or 1
    for i, (label, w) in enumerate(zip(labels, weights)):
        frac = w / total_weight
        cpu_spans.append({
            "label": label,
            "time_s": cpu_compute_wall * frac,
            "type": "compute",
            "estimated": True,
        })

    # Readback phase — measured directly
    cpu_spans.append({
        "label": "numpy readback",
        "time_s": cpu_readback_wall,
        "type": "readback",
        "estimated": False,
    })

    return {"fpga": fpga_spans, "cpu": cpu_spans}


def _flamegraph_svg(fg: dict, fmax_mhz: float | None) -> str:
    """Render flame graph spans as stacked horizontal SVG bars.

    FPGA spans are shown in cycles (and estimated wall time if fmax is known).
    CPU spans are shown in time units.
    """
    W = 700
    LEFT = 4
    RIGHT = 4
    BAR_H = 20
    GAP = 2
    TOP_PAD = 18
    LABEL_PAD = 4

    COLORS = {
        "load":     "#1f7a43",
        "compute":  "#d88412",
        "copy":     "#5a8fc4",
        "readback": "#a05cb0",
        "overhead": "#888888",
    }

    parts = []

    def _render_span_chart(title: str, spans: list, value_key: str,
                           domain: float, fmt_val, y_offset: int) -> int:
        """Render one set of spans; returns total height consumed."""
        bar_w = W - LEFT - RIGHT
        n = len(spans)
        chart_h = TOP_PAD + n * (BAR_H + GAP) + 4

        parts.append(f'<text x="{LEFT}" y="{y_offset + 13}" '
                     f'font-weight="bold" fill="#111" font-size="12">{title}</text>')

        for i, span in enumerate(spans):
            y = y_offset + TOP_PAD + i * (BAR_H + GAP)
            val = span[value_key]
            pct = (val / domain * 100) if domain > 0 else 0
            px = max(bar_w * val / domain, 1) if domain > 0 else 1

            color = COLORS.get(span["type"], "#999")
            est_marker = "\u2020" if span.get("estimated") else ""

            # Bar
            parts.append(
                f'<rect x="{LEFT}" y="{y}" width="{px:.1f}" height="{BAR_H}" '
                f'fill="{color}" rx="2">'
                f'<title>{html.escape(span["label"])}: {fmt_val(val)} ({pct:.1f}%)'
                f'{"  [estimated]" if span.get("estimated") else ""}</title>'
                f'</rect>'
            )

            # Label inside bar (if wide enough) or to the right
            label_text = f'{est_marker}{span["label"]}  {fmt_val(val)} ({pct:.1f}%)'
            text_x = LEFT + LABEL_PAD
            text_color = "#fff"
            if px < 180:
                text_x = LEFT + px + LABEL_PAD
                text_color = "#111"
            parts.append(
                f'<text x="{text_x:.1f}" y="{y + BAR_H - 5}" fill="{text_color}" '
                f'font-size="10" clip-path="url(#clip-fg)">{html.escape(label_text)}</text>'
            )

        return chart_h

    # --- FPGA chart ---
    fpga_spans = fg.get("fpga", [])
    fpga_total_cycles = sum(s["cycles"] for s in fpga_spans)

    def fmt_cycles(c):
        if fmax_mhz and fmax_mhz > 0:
            wall = c / (fmax_mhz * 1e6)
            m, u = _pick_unit(wall)
            return f"{c:,} cyc ({wall * m:.3f} {u})"
        return f"{c:,} cyc"

    fmax_note = f" @ {fmax_mhz:.1f} MHz" if fmax_mhz else " (no Fmax)"
    y = 0
    fpga_h = _render_span_chart(
        f"FPGA execution{fmax_note}  [\u2020 = estimated]",
        fpga_spans, "cycles", fpga_total_cycles, fmt_cycles, y,
    )
    y += fpga_h + 12

    # --- CPU chart ---
    cpu_spans = fg.get("cpu", [])
    cpu_total_s = sum(s["time_s"] for s in cpu_spans)
    mult_cpu, unit_cpu = _pick_unit(cpu_total_s)

    def fmt_time(t):
        return f"{t * mult_cpu:.4g} {unit_cpu}"

    cpu_h = _render_span_chart(
        f"CPU execution  [\u2020 = estimated]",
        cpu_spans, "time_s", cpu_total_s, fmt_time, y,
    )
    y += cpu_h

    total_h = y + 4
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" '
        f'style="font-family:monospace;font-size:11px;display:block;margin:4px 0 8px 0;">\n'
        f'<defs><clipPath id="clip-fg"><rect x="0" y="0" width="{W}" height="{total_h}"/>'
        f'</clipPath></defs>\n'
    )
    svg += "\n".join(parts)
    svg += "\n</svg>"
    return svg


# -- Estimates & Assumptions --

_ESTIMATES_AND_ASSUMPTIONS = [
    {
        "id": "analytical_cycle_model",
        "scope": "tg2hdl",
        "title": "Analytical cycle model",
        "description": (
            "Per-kernel cycle counts use _count_cycles_from_root() which assumes "
            "perfect pipelining with no memory stalls or pipeline bubbles. The real "
            "hardware FSM may differ due to wave grouping, prologue/epilogue states, "
            "and memory arbitration overhead."
        ),
    },
    {
        "id": "fsm_overhead_residual",
        "scope": "tg2hdl",
        "title": "FSM overhead (residual)",
        "description": (
            "Computed as the difference between observed simulation cycle count and "
            "the sum of analytical per-kernel + copy cycles. Lumps together start/done "
            "handshake, state-transition latency, and pipeline stalls into one bucket."
        ),
    },
    {
        "id": "copy_cycles_from_depth",
        "scope": "tg2hdl",
        "title": "Copy cycles = buffer depth",
        "description": (
            "Inter-kernel DMA copy cycles are set equal to the source buffer depth "
            "(one element per clock). Real copies may have burst setup latency or "
            "arbitration overhead not captured here."
        ),
    },
    {
        "id": "pcie_transfer_model",
        "scope": "tg2hdl",
        "title": "PCIe transfer time",
        "description": (
            "PCIe transfer times use a fixed bandwidth table (80% of theoretical "
            "link rate) and a constant 5 \u00b5s per-direction DMA latency. Real throughput "
            "depends on TLP payload size, MPS/MRRS settings, and host chipset."
        ),
    },
    {
        "id": "fpga_wall_from_fmax",
        "scope": "tg2hdl",
        "title": "FPGA wall time = cycles / Fmax",
        "description": (
            "FPGA wall time is derived by dividing total cycles by the synthesised "
            "Fmax. Actual deployed clock may differ due to voltage/temperature "
            "derating or user-selected clock constraints."
        ),
    },
    {
        "id": "cpu_per_item_proportional",
        "scope": "native",
        "title": "CPU per-item timing (proportional)",
        "description": (
            "tinygrad\u2019s run_schedule() executes the full schedule atomically, "
            "so per-schedule-item CPU time is unavailable. The flame graph distributes "
            "measured compute wall time proportionally by buffer byte size \u2014 a rough "
            "approximation that does not reflect actual per-kernel compute cost."
        ),
    },
]


def _render_html(report: dict, flamegraph_svg: str = "",
                 estimates: list | None = None) -> str:
    report_json = json.dumps(report)
    s = report["summary"]
    t = report["timing"]
    synth = report["top_synth"]["synth_stats"]

    mult, unit = _pick_unit(t["cpu_wall_s"], t.get("fpga_total_s"))
    timing_svg = _timing_svg(t)

    def tf(val):
        return _fmt_time(val, mult, unit) if val is not None else "n/a"

    status_style = "color:green;font-weight:bold" if s["correctness"] else "color:red;font-weight:bold"
    status_text  = "PASS" if s["correctness"] else "FAIL"

    # Per-kernel rows
    kernel_rows = ""
    for i, k in enumerate(report["kernels"]):
        ks = k["synth_stats"]
        fmax_str = "n/a" if ks.get("fmax_mhz") is None else f"{ks['fmax_mhz']:.1f}"
        name_str = k.get("metadata", [""])[0] if k.get("metadata") else ""
        kernel_rows += (
            f"<tr><td>{i}</td>"
            f"<td>{name_str}</td>"
            f"<td>{fmax_str}</td>"
            f"<td>{ks.get('comb','n/a')}</td><td>{ks.get('ff','n/a')}</td>"
            f"<td>{ks.get('dp16kd','n/a')}</td><td>{ks.get('mult18','n/a')}</td></tr>\n"
        )

    # KernelIR dumps
    kernelir_sections = ""
    for i, k in enumerate(report["kernels"]):
        ir = html.escape(k.get("kernel_ir") or "(not available)")
        kernelir_sections += f"<h3>Kernel {i}</h3>\n<pre>{ir}</pre>\n"

    # Schematics
    schematic_sections = ""
    top_svg = report["top_synth"].get("synth_svg") or ""
    if top_svg:
        schematic_sections += "<h3>TopModule</h3>\n" + top_svg + "\n"
    for i, k in enumerate(report["kernels"]):
        ksvg = k.get("synth_svg") or ""
        if ksvg:
            schematic_sections += f"<h3>Kernel {i}</h3>\n" + ksvg + "\n"
    if not schematic_sections:
        schematic_sections = "<p>No schematics available (Yosys/Graphviz not installed).</p>"

    # UOp viz divs
    uop_graph_divs = ""
    uop_graph_js = ""
    for i, k in enumerate(report["kernels"]):
        if k.get("tinygrad_graph"):
            uop_graph_divs += f'<h3>Kernel {i}</h3>\n<div class="graph" id="uop-graph-{i}"></div>\n'
            uop_graph_js += f'renderUOpGraph(document.getElementById("uop-graph-{i}"), REPORT.kernels[{i}].tinygrad_graph);\n'
    if not uop_graph_divs:
        uop_graph_divs = "<p>No UOp graphs available.</p>"

    ref_out  = html.escape(str(s["reference_output"]))
    sim_out  = html.escape(str(s["sim_output"]))

    fpga_time_note = "" if t["fpga_available"] else " (synthesis unavailable — cycles shown instead of time)"
    pm = t["pcie_model"]
    pcie_note = (
        f'PCIe Gen{pm["gen"]} x{pm["lanes"]} — '
        f'{pm["bw_gbs"]:.2f} GB/s practical, '
        f'{pm["latency_us"]:.1f} µs per-direction DMA latency'
    )

    # Estimates & Assumptions table rows
    est_list = estimates if estimates is not None else _ESTIMATES_AND_ASSUMPTIONS
    estimates_rows = ""
    for e in est_list:
        estimates_rows += (
            f'<tr><td>{html.escape(e["scope"])}</td>'
            f'<td>{html.escape(e["title"])}</td>'
            f'<td style="white-space:normal">{html.escape(e["description"])}</td></tr>\n'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>tg2hdl Report</title>
  <script src="assets/d3js.org/d3.v7.min.js"></script>
  <script src="assets/dagrejs.github.io/project/dagre/latest/dagre.min.js"></script>
  <style>
    body {{ font-family: monospace; margin: 32px 48px; background: #fff; color: #111; font-size: 13px; }}
    h1 {{ font-size: 16px; border-bottom: 2px solid #111; padding-bottom: 4px; margin-bottom: 16px; }}
    h2 {{ font-size: 14px; margin-top: 28px; margin-bottom: 6px; border-bottom: 1px solid #bbb; padding-bottom: 2px; }}
    h3 {{ font-size: 13px; margin-top: 14px; margin-bottom: 4px; }}
    table {{ border-collapse: collapse; margin: 6px 0 12px 0; }}
    th, td {{ border: 1px solid #bbb; padding: 3px 10px; text-align: left; white-space: nowrap; }}
    th {{ background: #f0f0f0; }}
    pre {{ background: #f8f8f8; border: 1px solid #ddd; padding: 8px 12px; font-size: 12px;
           overflow-x: auto; white-space: pre-wrap; word-break: break-word; margin: 4px 0 12px 0; }}
    .graph {{ width: 100%; height: 420px; border: 1px solid #bbb; margin: 6px 0 12px 0;
              position: relative; overflow: hidden; }}
    .graph svg {{ width: 100%; height: 100%; }}
    .graph-toolbar {{ position: absolute; top: 6px; right: 6px; display: flex; gap: 4px; z-index: 5; }}
    .graph-btn {{ border: 1px solid #bbb; background: #fff; padding: 2px 6px; cursor: pointer; font-size: 11px; }}
    .graph-detail {{ position: absolute; left: 8px; bottom: 8px; max-width: 320px; border: 1px solid #bbb;
                     background: rgba(255,255,255,0.97); padding: 6px 8px; font-size: 11px;
                     pointer-events: none; opacity: 0; white-space: pre-wrap; }}
    .graph-detail.visible {{ opacity: 1; }}
    .synth svg {{ width: 100%; height: auto; max-height: 600px; }}
    p {{ margin: 4px 0; }}
  </style>
</head>
<body>
  <h1>tg2hdl Report</h1>

  <h2>1. Summary</h2>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>Correctness</td><td style="{status_style}">{status_text}</td></tr>
    <tr><td>CPU device</td><td>{html.escape(s['tinygrad_device'])}</td></tr>
    <tr><td>FPGA target</td><td>{FPGA_FAMILY} {FPGA_DEVICE.upper()}-{FPGA_PACKAGE}</td></tr>
    <tr><td>Kernels</td><td>{len(report['kernels'])}</td></tr>
    <tr><td>Fmax</td><td>{_format_mhz(synth.get('fmax_mhz'))}</td></tr>
  </table>

  <h2>2. Timing{fpga_time_note}</h2>
  <p style="margin-bottom:6px">{pcie_note} &nbsp;|&nbsp; input {t['pcie_in_bytes']} B, output {t['pcie_out_bytes']} B</p>
  <table>
    <tr><th>Phase</th><th>CPU</th><th>FPGA cycles</th><th>FPGA (est.)</th></tr>
    <tr><td>PCIe in (host→FPGA)</td><td>—</td><td>—</td>
        <td>{tf(t.get('pcie_in_s'))}</td></tr>
    <tr><td>Input load (BRAM write)</td><td>—</td>
        <td>{t['load_cycles']}</td><td>{tf(t.get('fpga_load_s'))}</td></tr>
    <tr><td>Compute</td><td>{tf(t['cpu_compute_s'])}</td>
        <td>{t['compute_cycles']}</td><td>{tf(t.get('fpga_compute_s'))}</td></tr>
    <tr><td>Output readback (BRAM read)</td><td>{tf(t['cpu_readback_s'])}</td>
        <td>{t['readback_cycles']}</td><td>{tf(t.get('fpga_readback_s'))}</td></tr>
    <tr><td>PCIe out (FPGA→host)</td><td>—</td><td>—</td>
        <td>{tf(t.get('pcie_out_s'))}</td></tr>
    <tr><td><b>Total</b></td><td><b>{tf(t['cpu_wall_s'])}</b></td>
        <td><b>{t['total_cycles']}</b></td><td><b>{tf(t.get('fpga_with_pcie_s'))}</b></td></tr>
  </table>
  {timing_svg}

  <h2>3. FPGA Resources (full system)</h2>
  <table>
    <tr><th>Fmax</th><th>LUTs</th><th>FFs</th><th>BRAM (DP16KD)</th><th>DSPs (MULT18)</th><th>On-chip bits</th><th>Synth wall</th></tr>
    <tr>
      <td>{_format_mhz(synth.get('fmax_mhz'))}</td>
      <td>{synth.get('comb','n/a')}</td>
      <td>{synth.get('ff','n/a')}</td>
      <td>{synth.get('dp16kd','n/a')}</td>
      <td>{synth.get('mult18','n/a')}</td>
      <td>{synth.get('mem_bits','n/a')}</td>
      <td>{_format_seconds(synth.get('synth_wall_s'))}</td>
    </tr>
  </table>

  <h2>4. Per-Kernel Resources</h2>
  <table>
    <tr><th>#</th><th>Name</th><th>Fmax (MHz)</th><th>LUTs</th><th>FFs</th><th>BRAM</th><th>DSPs</th></tr>
    {kernel_rows}
  </table>

  <h2>5. Output Comparison</h2>
  <table>
    <tr><th>CPU reference</th><th>FPGA simulation</th></tr>
    <tr><td><pre style="margin:0;border:none;background:none">{ref_out}</pre></td>
        <td><pre style="margin:0;border:none;background:none">{sim_out}</pre></td></tr>
  </table>

  <h2>6. Kernel DAG</h2>
  <div class="graph" id="pipeline-graph"></div>

  <h2>7. Execution DAG</h2>
  <div class="graph" id="execution-graph"></div>

  <h2>8. KernelIR</h2>
  {kernelir_sections}

  <h2>9. Schematics</h2>
  <div class="synth">
  {schematic_sections}
  </div>

  <h2>10. Flame Graph</h2>
  <p style="margin-bottom:2px">Per-phase execution breakdown.  Color key:
  <span style="color:#1f7a43">■ load</span> &nbsp;
  <span style="color:#d88412">■ compute</span> &nbsp;
  <span style="color:#5a8fc4">■ copy/PCIe</span> &nbsp;
  <span style="color:#a05cb0">■ readback</span> &nbsp;
  <span style="color:#888">■ overhead</span> &nbsp;
  († = estimated value — see §11)
  </p>
  {flamegraph_svg}

  <h2>11. Estimates &amp; Assumptions</h2>
  <p>Values marked with † in the flame graph, and timing estimates throughout
  this report, rely on models documented below.  Each corresponds to a
  <code>FIXME</code> comment in the source code.</p>
  <table>
    <tr><th>Scope</th><th>Assumption</th><th>Description</th></tr>
    {estimates_rows}
  </table>

  <h2>12. tinygrad UOp Graphs</h2>
  {uop_graph_divs}

  <script>
    const REPORT = {report_json};

    // Strip ANSI escape codes from a string
    function stripAnsi(s) {{
      return (s || "").replace(/\x1b\[[0-9;]*m/g, "");
    }}

    // Build and render a dagre graph into container.
    // graphData: dict-of-dicts format (from graph_json or uop_to_json).
    //   Each key is a node id string; each value has: label, src (list of [edge_label, src_id]),
    //   color.  Labels may contain ANSI escapes and newlines.
    // labelLine: how many lines of the label to show in the node box (default 1).
    function _buildGraph(container, graphData, labelLine, nodeW, nodeH) {{
      nodeW = nodeW || 130;
      nodeH = nodeH || 36;
      labelLine = labelLine || 1;
      const root = d3.select(container);
      root.selectAll("*").remove();
      const toolbar = root.append("div").attr("class", "graph-toolbar");
      const detail = root.append("div").attr("class", "graph-detail");
      const svg = root.append("svg");
      const viewport = svg.append("g");
      const inner = viewport.append("g");
      const graph = new dagre.graphlib.Graph();
      graph.setGraph({{ rankdir: "TB", nodesep: 30, ranksep: 40 }});
      graph.setDefaultEdgeLabel(() => ({{}}));

      Object.entries(graphData).forEach(([id, n]) => {{
        const rawLabel = stripAnsi(n.label || id);
        const dispLabel = rawLabel.split("\\n").slice(0, labelLine).join(" | ");
        graph.setNode(String(id), {{
          label: dispLabel, fullLabel: rawLabel,
          width: nodeW, height: nodeH,
          color: n.color || "#e8edf2"
        }});
      }});
      Object.entries(graphData).forEach(([dstId, n]) => {{
        (n.src || []).forEach(e => {{
          const srcId = String(e[1]);
          const edgeLbl = typeof e[0] === "string" ? e[0] : "";
          if (graph.hasNode(srcId)) graph.setEdge(srcId, String(dstId), {{ label: edgeLbl }});
        }});
      }});

      dagre.layout(graph);
      const defs = svg.append("defs");
      const markId = "arr_" + container.id;
      defs.append("marker").attr("id", markId).attr("markerWidth",8).attr("markerHeight",6)
        .attr("refX",8).attr("refY",3).attr("orient","auto")
        .append("polygon").attr("points","0 0, 8 3, 0 6").attr("fill","#666");
      graph.edges().forEach(e => {{
        const pts = graph.edge(e).points;
        const path = d3.line().x(d=>d.x).y(d=>d.y).curve(d3.curveBasis)(pts);
        inner.append("path").attr("d",path).attr("fill","none")
          .attr("stroke","#666").attr("stroke-width",1.5).attr("marker-end","url(#"+markId+")");
        const ed = graph.edge(e);
        if (ed.label) {{
          const mp = pts[Math.floor(pts.length/2)];
          inner.append("text").attr("x",mp.x).attr("y",mp.y-4)
            .attr("text-anchor","middle").attr("font-size",9).attr("fill","#555").text(ed.label);
        }}
      }});
      graph.nodes().forEach(id => {{
        const n = graph.node(id);
        const g = inner.append("g").attr("transform",`translate(${{n.x-n.width/2}},${{n.y-n.height/2}})`);
        g.append("rect").attr("width",n.width).attr("height",n.height).attr("rx",3)
          .attr("fill",n.color).attr("stroke","#888").attr("stroke-width",1);
        g.append("text").attr("x",n.width/2).attr("y",n.height/2+4)
          .attr("text-anchor","middle").attr("font-size",9).attr("font-family","monospace")
          .text(n.label);
        g.on("click", () => {{
          detail.text(n.fullLabel).classed("visible", true);
          setTimeout(() => detail.classed("visible", false), 4000);
        }});
      }});
      const zoom = d3.zoom().scaleExtent([0.1,4]).on("zoom", ev => viewport.attr("transform",ev.transform));
      svg.call(zoom);
      const dims = graph.graph();
      const cw = container.clientWidth || 800, ch = container.clientHeight || 420;
      const sc = Math.min(cw/Math.max(dims.width+40,1), ch/Math.max(dims.height+40,1), 1);
      const tx = d3.zoomIdentity.translate((cw-dims.width*sc)/2+20,(ch-dims.height*sc)/2+20).scale(sc);
      svg.call(zoom.transform, tx);
      toolbar.append("button").attr("class","graph-btn").text("+").on("click",()=>svg.transition().call(zoom.scaleBy,1.3));
      toolbar.append("button").attr("class","graph-btn").text("−").on("click",()=>svg.transition().call(zoom.scaleBy,0.8));
      toolbar.append("button").attr("class","graph-btn").text("fit").on("click",()=>svg.transition().call(zoom.transform,tx));
    }}

    function renderGraph(container, graphData)    {{ _buildGraph(container, graphData, 1, 130, 36); }}
    function renderUOpGraph(container, graphData) {{ _buildGraph(container, graphData, 1, 110, 32); }}

    renderGraph(document.getElementById("pipeline-graph"),   REPORT.graphs.pipeline);
    renderGraph(document.getElementById("execution-graph"),  REPORT.graphs.execution);
    {uop_graph_js}
  </script>
</body>
</html>
"""

def benchmark(tensor, *extra_outputs, schedule_outputs=None, out_dir: str = "tg2hdl_report",
              pcie: PCIeModel | None = None) -> BenchmarkArtifact:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _copy_viz_assets(out_path)

    pcie = pcie or PCIeModel()

    if schedule_outputs is None:
        schedule = tensor.schedule(*extra_outputs)
    else:
        schedule_outputs = tuple(schedule_outputs)
        assert len(schedule_outputs) > 0, "schedule_outputs must not be empty"
        schedule = schedule_outputs[0].schedule(*schedule_outputs[1:])
    pipeline_view = analyze_schedule(schedule)
    top, _connections, kernel_specs = compile_top_module(schedule)
    input_data = _infer_input_data(schedule, pipeline_view, top)

    # Tinygrad reference — warm up first, then time compute and readback separately
    run_schedule(list(schedule), do_update_stats=False)

    t0 = time.perf_counter()
    run_schedule(list(schedule), do_update_stats=False)
    cpu_compute_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    ref_out = np.array(tensor.numpy(), copy=True).reshape(-1)
    cpu_readback_wall = time.perf_counter() - t1

    tinygrad_wall = cpu_compute_wall + cpu_readback_wall

    sim_out_raw, cycle_counts, sim_wall = simulate_top(top, input_data)
    sim_out = _decode_sim_output(sim_out_raw, tensor.dtype).reshape(-1)

    if np.issubdtype(ref_out.dtype, np.floating):
        correctness = bool(np.allclose(ref_out.reshape(-1), sim_out.reshape(-1), rtol=1e-5, atol=1e-5))
    else:
        correctness = bool(np.array_equal(ref_out.reshape(-1), sim_out.reshape(-1)))

    # PCIe transfer times
    input_bytes  = sum(arr.nbytes for arr in input_data.values())
    output_bytes = sim_out_raw.nbytes
    pcie_in_s    = pcie.xfer_s(input_bytes)
    pcie_out_s   = pcie.xfer_s(output_bytes)

    report = {
        "summary": {
            "tinygrad_device": str(tensor.device),
            "tinygrad_wall_s": tinygrad_wall,
            "tg2hdl_wall_s": sim_wall,
            "tg2hdl_cycles": cycle_counts["compute"],
            "tg2hdl_load_cycles": cycle_counts["load"],
            "tg2hdl_readback_cycles": cycle_counts["readback"],
            "tg2hdl_total_cycles": cycle_counts["total"],
            "correctness": correctness,
            "reference_output": ref_out.reshape(-1).tolist(),
            "sim_output": sim_out.reshape(-1).tolist(),
        },
        "graphs": _graph_payload(pipeline_view),
        "top_synth": _top_payload(top, out_path / "synth"),
        "kernels": _kernel_payload(schedule, pipeline_view, kernel_specs, out_path / "synth"),
    }

    fmax_mhz = report["top_synth"]["synth_stats"]["fmax_mhz"]
    total_cycles = cycle_counts["total"]
    fpga_wall_s = (total_cycles / (fmax_mhz * 1e6)) if (fmax_mhz and fmax_mhz > 0) else None
    fpga_with_pcie_s = (fpga_wall_s + pcie_in_s + pcie_out_s) if fpga_wall_s is not None else None

    def _cycles_to_s(c):
        return (c / (fmax_mhz * 1e6)) if (fmax_mhz and fmax_mhz > 0) else None

    report["timing"] = {
        "cpu_compute_s":    cpu_compute_wall,
        "cpu_readback_s":   cpu_readback_wall,
        "cpu_wall_s":       tinygrad_wall,
        "fpga_available":   fpga_wall_s is not None,
        "fpga_load_s":      _cycles_to_s(cycle_counts["load"]),
        "fpga_compute_s":   _cycles_to_s(cycle_counts["compute"]),
        "fpga_readback_s":  _cycles_to_s(cycle_counts["readback"]),
        "fpga_total_s":     fpga_wall_s,
        "load_cycles":      cycle_counts["load"],
        "compute_cycles":   cycle_counts["compute"],
        "readback_cycles":  cycle_counts["readback"],
        "total_cycles":     total_cycles,
        # PCIe model
        "pcie_in_s":        pcie_in_s,
        "pcie_out_s":       pcie_out_s,
        "fpga_with_pcie_s": fpga_with_pcie_s,
        "pcie_in_bytes":    input_bytes,
        "pcie_out_bytes":   output_bytes,
        "pcie_model": {
            "gen":        pcie.gen,
            "lanes":      pcie.lanes,
            "latency_us": pcie.latency_s * 1e6,
            "bw_gbs":     pcie.bw_bytes_s / 1e9,
        },
    }

    # Flame graph
    fg = _flamegraph_payload(
        schedule, pipeline_view, top, kernel_specs,
        cycle_counts, tinygrad_wall, cpu_compute_wall, cpu_readback_wall,
    )
    fg_svg = _flamegraph_svg(fg, fmax_mhz)

    html_path = out_path / "index.html"
    html_path.write_text(_render_html(report, flamegraph_svg=fg_svg,
                                      estimates=_ESTIMATES_AND_ASSUMPTIONS))
    return BenchmarkArtifact(
        output_dir=str(out_path),
        report_path=str(html_path),
        tinygrad_device=str(tensor.device),
        tinygrad_wall_s=tinygrad_wall,
        tg2hdl_wall_s=sim_wall,
        tg2hdl_cycles=cycle_counts["compute"],
        tg2hdl_total_cycles=total_cycles,
        correctness=correctness,
        fpga_wall_s=fpga_wall_s,
        pcie_in_s=pcie_in_s,
        pcie_out_s=pcie_out_s,
        fpga_with_pcie_s=fpga_with_pcie_s,
    )
