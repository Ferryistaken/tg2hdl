from __future__ import annotations

import html
import json
import re
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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _format_metadata(meta) -> str:
    names = [str(m.name if hasattr(m, "name") else m) for m in meta]
    return f" ({', '.join(names)})" if names else ""


def _measure_cpu_schedule(schedule) -> tuple[list[dict], float]:
    spans = []
    total = 0.0
    compute_idx = 0
    copy_idx = 0

    for si in schedule:
        ei = si.lower()
        t0 = time.perf_counter()
        et = ei.run(wait=True, do_update_stats=False)
        dt = et if et is not None else (time.perf_counter() - t0)
        total += dt

        if si.ast.op == Ops.SINK:
            label = f"K{compute_idx}{_format_metadata(getattr(si, 'metadata', ()))}"
            span_type = "compute"
            compute_idx += 1
        elif si.ast.op == Ops.COPY:
            label = f"copy {copy_idx}"
            span_type = "copy"
            copy_idx += 1
        else:
            label = _strip_ansi(getattr(getattr(ei, "prg", None), "display_name", si.ast.op.name)).strip() or si.ast.op.name
            span_type = "overhead"

        spans.append({
            "label": label,
            "time_s": dt,
            "type": span_type,
            "estimated": False,
            "detail": _strip_ansi(getattr(getattr(ei, "prg", None), "display_name", "")).strip(),
        })

    return spans, total


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

def _fg_node(label: str, value: float, node_type: str, *, estimated: bool = False,
             detail: str = "", children: list[dict] | None = None) -> dict:
    return {
        "label": label,
        "value": value,
        "type": node_type,
        "estimated": estimated,
        "detail": detail,
        "children": list(children or []),
    }


def _resource_summary(stats: dict | None) -> str:
    if not stats:
        return ""
    parts = []
    fmax = stats.get("fmax_mhz")
    if fmax is not None:
        parts.append(f"Fmax {fmax:.1f} MHz")
    for key, label in (("comb", "LUT"), ("ff", "FF"), ("dp16kd", "BRAM"), ("mult18", "DSP")):
        if stats.get(key) is not None:
            parts.append(f"{label} {stats[key]}")
    return ", ".join(parts)


def _flamegraph_payload(schedule, pipeline_view, top, cycle_counts: dict, cpu_spans: list[dict],
                        cpu_readback_wall: float, timing: dict, kernels: list[dict],
                        fmax_mhz: float | None) -> dict:
    """Build hierarchical flame graph data for FPGA and CPU."""
    compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
    # FIXME: When Fmax is present, the FPGA flame graph uses derived wall time
    # (cycles / synthesized Fmax), not a directly measured board-level runtime.
    state_cycles = cycle_counts.get("states", {})
    fmax_hz = (fmax_mhz * 1e6) if (fmax_mhz and fmax_mhz > 0) else None

    def cyc_to_val(cycles: int) -> float:
        return (cycles / fmax_hz) if fmax_hz else float(cycles)

    def cyc_detail(cycles: int) -> str:
        if fmax_hz:
            m, u = _pick_unit(cycles / fmax_hz)
            return f"{cycles} cyc, {(cycles / fmax_hz) * m:.4g} {u}"
        return f"{cycles} cyc"

    fpga_compute_children = []
    measured_compute_cycles = 0
    for exec_k, orig_k in enumerate(pipeline_view.execution_kernel_map):
        si = compute_items[orig_k]
        run_cycles = state_cycles.get(f"K{exec_k}_RUN", 0)
        wait_cycles = state_cycles.get(f"K{exec_k}_WAIT", 0)
        kernel_cycles = run_cycles + wait_cycles
        measured_compute_cycles += kernel_cycles
        kstats = kernels[exec_k]["synth_stats"] if exec_k < len(kernels) else {}
        kchildren = []
        if run_cycles > 0:
            kchildren.append(_fg_node("RUN", cyc_to_val(run_cycles), "compute", detail=cyc_detail(run_cycles)))
        if wait_cycles > 0:
            kchildren.append(_fg_node("WAIT", cyc_to_val(wait_cycles), "overhead", detail=cyc_detail(wait_cycles)))
        fpga_compute_children.append(_fg_node(
            f"K{exec_k}{_format_metadata(si.metadata)}",
            cyc_to_val(kernel_cycles),
            "compute",
            detail=_resource_summary(kstats),
            children=kchildren,
        ))

        for gi, group in enumerate(top._copy_groups.get(exec_k, [])):
            src_buf = group[0][1]
            dsts = [f"K{dk}.b{db}" for _, _, dk, db in group]
            copy_cycles = state_cycles.get(f"K{exec_k}_COPY_{gi}", 0)
            measured_compute_cycles += copy_cycles
            fpga_compute_children.append(_fg_node(
                f"copy K{exec_k}.b{src_buf}\u2192{','.join(dsts)}",
                cyc_to_val(copy_cycles),
                "copy",
                detail=cyc_detail(copy_cycles),
            ))

    fpga_overhead_cycles = max(0, cycle_counts["compute"] - measured_compute_cycles)
    if fpga_overhead_cycles > 0:
        fpga_compute_children.append(_fg_node(
            "FSM/control overhead",
            cyc_to_val(fpga_overhead_cycles),
            "overhead",
            detail=cyc_detail(fpga_overhead_cycles),
        ))

    fpga_in_children = []
    if timing.get("pcie_in_s") is not None and fmax_hz:
        # FIXME: PCIe transfer leaves are still model-based, not measured from a
        # real PCIe-enabled FPGA DMA path or hardware timestamp source.
        fpga_in_children.append(_fg_node(
            "PCIe host→FPGA",
            timing["pcie_in_s"],
            "copy",
            estimated=True,
            detail=f"{timing['pcie_in_bytes']} B",
        ))
    fpga_in_children.append(_fg_node(
        "BRAM load",
        cyc_to_val(cycle_counts["load"]),
        "load",
        detail=cyc_detail(cycle_counts["load"]),
    ))

    fpga_out_children = [
        _fg_node("BRAM readback", cyc_to_val(cycle_counts["readback"]), "readback",
                 detail=cyc_detail(cycle_counts["readback"]))
    ]
    if timing.get("pcie_out_s") is not None and fmax_hz:
        # FIXME: PCIe transfer leaves are still model-based, not measured from a
        # real PCIe-enabled FPGA DMA path or hardware timestamp source.
        fpga_out_children.append(_fg_node(
            "PCIe FPGA→host",
            timing["pcie_out_s"],
            "copy",
            estimated=True,
            detail=f"{timing['pcie_out_bytes']} B",
        ))

    fpga_phase_nodes = [
        _fg_node("Input transfer", sum(c["value"] for c in fpga_in_children), "load", children=fpga_in_children),
        _fg_node("Compute", sum(c["value"] for c in fpga_compute_children), "compute", children=fpga_compute_children),
        _fg_node("Output transfer", sum(c["value"] for c in fpga_out_children), "readback", children=fpga_out_children),
    ]
    fpga_total = _fg_node(
        "FPGA total",
        sum(c["value"] for c in fpga_phase_nodes),
        "overhead",
        children=[c for c in fpga_phase_nodes if c["value"] > 0],
    )

    cpu_input_children = [span for span in cpu_spans if span["type"] == "copy"]
    cpu_compute_children = [span for span in cpu_spans if span["type"] == "compute"]
    cpu_overhead_children = [span for span in cpu_spans if span["type"] not in {"copy", "compute"}]

    cpu_phase_nodes = []
    if cpu_input_children:
        cpu_phase_nodes.append(_fg_node(
            "Input transfer",
            sum(s["time_s"] for s in cpu_input_children),
            "copy",
            children=[_fg_node(s["label"], s["time_s"], s["type"], detail=s.get("detail", "")) for s in cpu_input_children],
        ))
    if cpu_compute_children:
        cpu_phase_nodes.append(_fg_node(
            "Compute",
            sum(s["time_s"] for s in cpu_compute_children),
            "compute",
            children=[_fg_node(s["label"], s["time_s"], s["type"], detail=s.get("detail", "")) for s in cpu_compute_children],
        ))
    if cpu_overhead_children:
        cpu_phase_nodes.append(_fg_node(
            "Runtime overhead",
            sum(s["time_s"] for s in cpu_overhead_children),
            "overhead",
            children=[_fg_node(s["label"], s["time_s"], s["type"], detail=s.get("detail", "")) for s in cpu_overhead_children],
        ))

    cpu_out_children = [_fg_node("numpy readback", cpu_readback_wall, "readback")]
    cpu_phase_nodes.append(_fg_node("Output transfer", cpu_readback_wall, "readback", children=cpu_out_children))
    cpu_total = _fg_node("CPU total", sum(c["value"] for c in cpu_phase_nodes), "overhead", children=cpu_phase_nodes)

    unit = "time_s" if fmax_hz else "cycles"
    return {"fpga": fpga_total, "cpu": cpu_total, "unit": unit}


def _flamegraph_svg(fg: dict, fmax_mhz: float | None) -> str:
    """Render hierarchical flame graphs with independent zoomable panels."""
    W = 1040
    LEFT = 8
    RIGHT = 8
    TITLE_H = 18
    BAR_H = 28
    LEVEL_GAP = 3
    TEXT_PAD = 4
    bar_w = W - LEFT - RIGHT

    BASE_COLORS = {
        "load": (0x53, 0x9b, 0x4b),
        "compute": (0xe2, 0x86, 0x2b),
        "copy": (0x63, 0x97, 0xc5),
        "readback": (0xb2, 0x61, 0xb9),
        "overhead": (0x8b, 0x8b, 0x8b),
    }

    def _mix_color(rgb: tuple[int, int, int], factor: float) -> str:
        factor = max(-0.35, min(0.35, factor))
        if factor >= 0:
            vals = [int(c + (255 - c) * factor) for c in rgb]
        else:
            vals = [int(c * (1 + factor)) for c in rgb]
        return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"

    def _span_fill(node: dict) -> str:
        base = BASE_COLORS.get(node["type"], (0x99, 0x99, 0x99))
        jitter = (((sum(ord(c) for c in node["label"]) % 13) - 6) / 30.0)
        return _mix_color(base, jitter)

    def _trim_label(text: str, width_px: float) -> str:
        approx_chars = max(int((width_px - 2 * TEXT_PAD) / 6.7), 0)
        if approx_chars <= 0:
            return ""
        if len(text) <= approx_chars:
            return text
        if approx_chars <= 3:
            return ""
        return text[:approx_chars - 1] + "…"

    def _max_depth(node: dict) -> int:
        kids = node.get("children") or []
        return 1 if not kids else 1 + max(_max_depth(child) for child in kids)

    def _render_node(node: dict, x: float, width: float, depth_from_root: int, max_depth: int,
                     fmt_val, domain: float, y_base: int, parts: list[str], clip_id: str) -> None:
        y = y_base + (max_depth - depth_from_root - 1) * (BAR_H + LEVEL_GAP)
        pct = (node["value"] / domain * 100) if domain > 0 else 0
        est_note = " [estimated]" if node.get("estimated") else ""
        tooltip = f'{node["label"]}: {fmt_val(node["value"])} ({pct:.1f}%){est_note}'
        if node.get("detail"):
            tooltip += f" | {node['detail']}"
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(width, 1):.2f}" height="{BAR_H}" '
            f'fill="{_span_fill(node)}" stroke="#fff7eb" stroke-width="1">'
            f'<title>{html.escape(tooltip)}</title></rect>'
        )
        text = _trim_label(node["label"], width)
        if text:
            parts.append(
                f'<text x="{x + TEXT_PAD:.2f}" y="{y + 18:.2f}" fill="#24170b" font-size="10" '
                f'clip-path="url(#{clip_id})">{html.escape(text)}</text>'
            )

        children = [child for child in (node.get("children") or []) if child["value"] > 0]
        if not children:
            return

        cursor = x
        total_child_value = sum(child["value"] for child in children)
        for idx, child in enumerate(children):
            child_width = width * child["value"] / total_child_value if total_child_value > 0 else 0
            if idx == len(children) - 1:
                child_width = (x + width) - cursor
            _render_node(child, cursor, child_width, depth_from_root + 1, max_depth, fmt_val, domain, y_base, parts, clip_id)
            cursor += child_width

    def _render_chart_svg(title: str, root: dict, fmt_val, chart_id: str) -> str:
        max_depth = _max_depth(root)
        chart_h = TITLE_H + max_depth * BAR_H + (max_depth - 1) * LEVEL_GAP + 4
        clip_id = f"clip-fg-{chart_id}"
        parts: list[str] = [
            f'<text x="{LEFT}" y="13" font-weight="bold" fill="#111" font-size="12">{html.escape(title)}</text>'
        ]
        _render_node(root, LEFT, bar_w, 0, max_depth, fmt_val, root["value"], TITLE_H, parts, clip_id)
        svg = (
            f'<svg class="flamegraph-svg" xmlns="http://www.w3.org/2000/svg" width="{W}" height="{chart_h}" '
            f'data-base-width="{W}" style="font-family:monospace;font-size:11px;display:block;margin:0;">\n'
            f'<defs><clipPath id="{clip_id}"><rect x="0" y="0" width="{W * 8}" height="{chart_h}"/></clipPath></defs>\n'
        )
        svg += f'<g class="flamegraph-root" transform="scale(1,1)">\n'
        svg += "\n".join(parts)
        svg += "\n</g>"
        svg += "\n</svg>"
        return svg

    def fmt_fpga(v: float) -> str:
        if fg.get("unit") == "time_s":
            mult, unit = _pick_unit(v)
            return f"{v * mult:.4g} {unit}"
        return f"{int(round(v))} cyc"

    cpu_total = fg["cpu"]["value"]
    mult_cpu, unit_cpu = _pick_unit(cpu_total)

    def fmt_cpu(v: float) -> str:
        return f"{v * mult_cpu:.4g} {unit_cpu}"

    fmax_note = f" @ {fmax_mhz:.1f} MHz" if (fmax_mhz and fg.get("unit") == "time_s") else ""
    fpga_svg = _render_chart_svg(f"FPGA execution{fmax_note}", fg["fpga"], fmt_fpga, "fpga")
    cpu_svg = _render_chart_svg("CPU execution", fg["cpu"], fmt_cpu, "cpu")

    return (
        '<div class="flamegraph-panel" data-default-zoom="2.5">'
        '<div class="flamegraph-controls">'
        '<button type="button" class="flamegraph-btn" data-zoom-step="-0.25">-</button>'
        '<input class="flamegraph-zoom" type="range" min="1" max="8" step="0.25" value="2.5" />'
        '<button type="button" class="flamegraph-btn" data-zoom-step="0.25">+</button>'
        '<span class="flamegraph-zoom-label">2.50x</span>'
        '</div>'
        f'<div class="flamegraph-scroll">{fpga_svg}</div>'
        '</div>'
        '<div class="flamegraph-panel" data-default-zoom="1.25">'
        '<div class="flamegraph-controls">'
        '<button type="button" class="flamegraph-btn" data-zoom-step="-0.25">-</button>'
        '<input class="flamegraph-zoom" type="range" min="1" max="8" step="0.25" value="1.25" />'
        '<button type="button" class="flamegraph-btn" data-zoom-step="0.25">+</button>'
        '<span class="flamegraph-zoom-label">1.25x</span>'
        '</div>'
        f'<div class="flamegraph-scroll">{cpu_svg}</div>'
        '</div>'
    )


# -- Estimates & Assumptions --

_ESTIMATES_AND_ASSUMPTIONS = [
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

    top_svg = report["top_synth"].get("synth_svg") or ""
    top_schematic = top_svg if top_svg else "<p>No top-module schematic available (Yosys/Graphviz not installed).</p>"

    kernel_sections = ""
    for i, k in enumerate(report["kernels"]):
        ks = k["synth_stats"]
        meta = ", ".join(k.get("metadata") or []) or "unnamed"
        ir = html.escape(k.get("kernel_ir") or "(not available)")
        ksvg = k.get("synth_svg") or "<p>No schematic available.</p>"
        uop_div = (
            f'<div class="graph kernel-uop-graph" id="uop-graph-{i}" data-kernel-index="{i}"></div>'
            if k.get("tinygrad_graph") else
            "<p>No UOp graph available.</p>"
        )
        kernel_sections += f"""
  <details class="kernel-detail">
    <summary>Kernel {i}: {html.escape(meta)}</summary>
    <h3>Kernel {i} Resources</h3>
    <table>
      <tr><th>Fmax (MHz)</th><th>LUTs</th><th>FFs</th><th>BRAM</th><th>DSPs</th></tr>
      <tr>
        <td>{'n/a' if ks.get('fmax_mhz') is None else f"{ks['fmax_mhz']:.1f}"}</td>
        <td>{ks.get('comb','n/a')}</td>
        <td>{ks.get('ff','n/a')}</td>
        <td>{ks.get('dp16kd','n/a')}</td>
        <td>{ks.get('mult18','n/a')}</td>
      </tr>
    </table>
    <h3>UOp Graph</h3>
    {uop_div}
    <h3>KernelIR</h3>
    <pre>{ir}</pre>
    <h3>Schematic</h3>
    <div class="synth">{ksvg}</div>
  </details>
"""
    if not kernel_sections:
        kernel_sections = "<p>No kernel details available.</p>"

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
    h4 {{ font-size: 12px; margin-top: 10px; margin-bottom: 4px; }}
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
    .flamegraph-panel {{ margin: 8px 0 14px 0; }}
    .flamegraph-controls {{ display: flex; align-items: center; gap: 8px; margin: 0 0 6px 0; }}
    .flamegraph-btn {{ border: 1px solid #bbb; background: #fff; padding: 2px 8px; cursor: pointer; font-size: 11px; }}
    .flamegraph-zoom {{ width: 180px; }}
    .flamegraph-zoom-label {{ min-width: 42px; }}
    .flamegraph-scroll {{ overflow-x: auto; overflow-y: hidden; border: 1px solid #bbb; background: #faf8f1; }}
    .flamegraph-scroll svg {{ display: block; }}
    details.kernel-detail {{ margin: 10px 0 14px 0; border: 1px solid #ccc; background: #fbfbfb; }}
    details.kernel-detail > summary {{ cursor: pointer; padding: 8px 10px; font-weight: bold; background: #f3f3f3; }}
    details.kernel-detail[open] > summary {{ border-bottom: 1px solid #ccc; }}
    details.kernel-detail > :not(summary) {{ margin-left: 10px; margin-right: 10px; }}
    p {{ margin: 4px 0; }}
  </style>
</head>
<body>
  <h1>tg2hdl Report</h1>

  <h2>1. Summary</h2>
  <h3>Run Summary</h3>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>Correctness</td><td style="{status_style}">{status_text}</td></tr>
    <tr><td>CPU device</td><td>{html.escape(s['tinygrad_device'])}</td></tr>
    <tr><td>FPGA target</td><td>{FPGA_FAMILY} {FPGA_DEVICE.upper()}-{FPGA_PACKAGE}</td></tr>
    <tr><td>Kernels</td><td>{len(report['kernels'])}</td></tr>
    <tr><td>Fmax</td><td>{_format_mhz(synth.get('fmax_mhz'))}</td></tr>
  </table>

  <h3>Output Comparison</h3>
  <table>
    <tr><th>CPU reference</th><th>FPGA simulation</th></tr>
    <tr><td><pre style="margin:0;border:none;background:none">{ref_out}</pre></td>
        <td><pre style="margin:0;border:none;background:none">{sim_out}</pre></td></tr>
  </table>

  <h2>2. Performance</h2>
  <h3>Timing{fpga_time_note}</h3>
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

  <h3>Flame Graph</h3>
  <p style="margin-bottom:2px">Inclusive flame graph with total cost at the bottom and hierarchical breakdown above. Color key:
  <span style="color:#1f7a43">■ load</span> &nbsp;
  <span style="color:#d88412">■ compute</span> &nbsp;
  <span style="color:#5a8fc4">■ copy/PCIe</span> &nbsp;
  <span style="color:#a05cb0">■ readback</span> &nbsp;
  <span style="color:#888">■ overhead</span>
  </p>
  <div class="flamegraph-panel" id="flamegraph-fpga-panel" data-kind="fpga" data-default-zoom="2.5">
    <div class="flamegraph-controls">
      <button type="button" class="flamegraph-btn" data-zoom-step="-0.25">-</button>
      <input class="flamegraph-zoom" type="range" min="1" max="8" step="0.25" value="2.5" />
      <button type="button" class="flamegraph-btn" data-zoom-step="0.25">+</button>
      <span class="flamegraph-zoom-label">2.50x</span>
    </div>
    <div class="flamegraph-scroll"></div>
  </div>
  <div class="flamegraph-panel" id="flamegraph-cpu-panel" data-kind="cpu" data-default-zoom="1.25">
    <div class="flamegraph-controls">
      <button type="button" class="flamegraph-btn" data-zoom-step="-0.25">-</button>
      <input class="flamegraph-zoom" type="range" min="1" max="8" step="0.25" value="1.25" />
      <button type="button" class="flamegraph-btn" data-zoom-step="0.25">+</button>
      <span class="flamegraph-zoom-label">1.25x</span>
    </div>
    <div class="flamegraph-scroll"></div>
  </div>

  <h2>3. Resources</h2>
  <h3>Full System Resources</h3>
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

  <h3>Per-Kernel Resources</h3>
  <table>
    <tr><th>#</th><th>Name</th><th>Fmax (MHz)</th><th>LUTs</th><th>FFs</th><th>BRAM</th><th>DSPs</th></tr>
    {kernel_rows}
  </table>

  <h3>Top Module Schematic</h3>
  <div class="synth">
  {top_schematic}
  </div>

  <h2>4. Kernels</h2>
  <h3>Execution Views</h3>
  <h4>Kernel DAG</h4>
  <div class="graph" id="pipeline-graph"></div>
  <h4>Execution DAG</h4>
  <div class="graph" id="execution-graph"></div>
  <h3>Kernel Details</h3>
  <p>Kernel details are collapsed by default.</p>
  {kernel_sections}

  <h2>5. Estimates</h2>
  <p>Measured flame graph spans are shown directly above. Timing estimates that
  still exist elsewhere in the report are documented below. Each corresponds to
  a <code>FIXME</code> comment in the source code.</p>
  <table>
    <tr><th>Scope</th><th>Assumption</th><th>Description</th></tr>
    {estimates_rows}
  </table>

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

    const FLAMEGRAPH_COLORS = {{
      load: [0x53, 0x9b, 0x4b],
      compute: [0xe2, 0x86, 0x2b],
      copy: [0x63, 0x97, 0xc5],
      readback: [0xb2, 0x61, 0xb9],
      overhead: [0x8b, 0x8b, 0x8b],
    }};

    function flamegraphMixColor(rgb, factor) {{
      const clamped = Math.max(-0.35, Math.min(0.35, factor));
      const vals = rgb.map(c => clamped >= 0 ? Math.round(c + (255 - c) * clamped) : Math.round(c * (1 + clamped)));
      return `#${{vals.map(v => v.toString(16).padStart(2, "0")).join("")}}`;
    }}

    function flamegraphFill(node) {{
      const base = FLAMEGRAPH_COLORS[node.type] || [0x99, 0x99, 0x99];
      const hash = Array.from(node.label || "").reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
      const jitter = ((hash % 13) - 6) / 30;
      return flamegraphMixColor(base, jitter);
    }}

    function flamegraphDepth(node) {{
      const kids = node.children || [];
      return kids.length ? 1 + Math.max(...kids.map(flamegraphDepth)) : 1;
    }}

    function flamegraphTrim(text, widthPx) {{
      const chars = Math.max(Math.floor((widthPx - 8) / 6.7), 0);
      if (chars <= 0) return "";
      if (text.length <= chars) return text;
      if (chars <= 3) return "";
      return text.slice(0, chars - 1) + "…";
    }}

    function flamegraphPickUnit(sec) {{
      if (sec < 1e-6) return [1e9, "ns"];
      if (sec < 1e-3) return [1e6, "µs"];
      if (sec < 1) return [1e3, "ms"];
      return [1, "s"];
    }}

    function flamegraphFormatValue(value, unit) {{
      if (unit === "cycles") return `${{Math.round(value)}} cyc`;
      const [mult, label] = flamegraphPickUnit(value);
      return `${{(value * mult).toPrecision(4)}} ${{label}}`;
    }}

    function renderFlamegraphPanel(panel, rootNode, title, unit) {{
      const BAR_H = 28, LEVEL_GAP = 3, TITLE_H = 18, LEFT = 8, RIGHT = 8, TEXT_PAD = 4;
      const slider = panel.querySelector(".flamegraph-zoom");
      const label = panel.querySelector(".flamegraph-zoom-label");
      const scroll = panel.querySelector(".flamegraph-scroll");
      const panelWidth = Math.max(scroll.clientWidth || 0, 1040);
      const zoom = Math.max(1, Number(slider.value) || 1);
      const svgWidth = panelWidth * zoom;
      const maxDepth = flamegraphDepth(rootNode);
      const svgHeight = TITLE_H + maxDepth * BAR_H + (maxDepth - 1) * LEVEL_GAP + 4;
      const usableWidth = svgWidth - LEFT - RIGHT;

      const svg = d3.create("svg")
        .attr("xmlns", "http://www.w3.org/2000/svg")
        .attr("width", svgWidth)
        .attr("height", svgHeight)
        .style("font-family", "monospace")
        .style("font-size", "11px")
        .style("display", "block")
        .style("margin", 0);

      svg.append("text")
        .attr("x", LEFT)
        .attr("y", 13)
        .attr("font-weight", "bold")
        .attr("fill", "#111")
        .attr("font-size", 12)
        .text(title);

      function drawNode(node, x, width, depthFromRoot) {{
        const y = TITLE_H + (maxDepth - depthFromRoot - 1) * (BAR_H + LEVEL_GAP);
        const pct = rootNode.value > 0 ? (node.value / rootNode.value * 100) : 0;
        let tip = `${{node.label}}: ${{flamegraphFormatValue(node.value, unit)}} (${{pct.toFixed(1)}}%)`;
        if (node.estimated) tip += " [estimated]";
        if (node.detail) tip += ` | ${{node.detail}}`;

        svg.append("rect")
          .attr("x", x)
          .attr("y", y)
          .attr("width", Math.max(width, 1))
          .attr("height", BAR_H)
          .attr("fill", flamegraphFill(node))
          .attr("stroke", "#fff7eb")
          .attr("stroke-width", 1)
          .append("title")
          .text(tip);

        const text = flamegraphTrim(node.label, width);
        if (text) {{
          svg.append("text")
            .attr("x", x + TEXT_PAD)
            .attr("y", y + 18)
            .attr("fill", "#24170b")
            .attr("font-size", 10)
            .text(text);
        }}

        const children = (node.children || []).filter(child => child.value > 0);
        if (!children.length) return;
        const total = children.reduce((acc, child) => acc + child.value, 0);
        let cursor = x;
        children.forEach((child, idx) => {{
          let childWidth = total > 0 ? width * child.value / total : 0;
          if (idx === children.length - 1) childWidth = x + width - cursor;
          drawNode(child, cursor, childWidth, depthFromRoot + 1);
          cursor += childWidth;
        }});
      }}

      drawNode(rootNode, LEFT, usableWidth, 0);
      scroll.replaceChildren(svg.node());
      label.textContent = `${{zoom.toFixed(2)}}x`;
    }}

    function initFlamegraphPanel(panel, rootNode, title, unit) {{
      const slider = panel.querySelector(".flamegraph-zoom");
      const scroll = panel.querySelector(".flamegraph-scroll");
      const render = () => renderFlamegraphPanel(panel, rootNode, title, unit);

      panel.querySelectorAll("[data-zoom-step]").forEach(btn => {{
        btn.addEventListener("click", () => {{
          const step = Number(btn.getAttribute("data-zoom-step")) || 0;
          const next = Math.min(Number(slider.max), Math.max(Number(slider.min), Number(slider.value) + step));
          slider.value = next.toFixed(2);
          render();
          scroll.focus();
        }});
      }});

      slider.addEventListener("input", render);
      render();
      window.addEventListener("resize", render);
    }}

    initFlamegraphPanel(
      document.getElementById("flamegraph-fpga-panel"),
      REPORT.flamegraph.fpga,
      REPORT.flamegraph.unit === "time_s" && REPORT.top_synth.synth_stats.fmax_mhz
        ? `FPGA execution @ ${{REPORT.top_synth.synth_stats.fmax_mhz.toFixed(1)}} MHz`
        : "FPGA execution",
      REPORT.flamegraph.unit
    );
    initFlamegraphPanel(
      document.getElementById("flamegraph-cpu-panel"),
      REPORT.flamegraph.cpu,
      "CPU execution",
      "time_s"
    );

    function renderKernelGraphIfNeeded(idx) {{
      const el = document.getElementById(`uop-graph-${{idx}}`);
      if (!el || el.dataset.rendered === "1") return;
      renderUOpGraph(el, REPORT.kernels[idx].tinygrad_graph);
      el.dataset.rendered = "1";
    }}

    document.querySelectorAll("details.kernel-detail").forEach(detail => {{
      detail.addEventListener("toggle", () => {{
        if (!detail.open) return;
        detail.querySelectorAll(".kernel-uop-graph[data-kernel-index]").forEach(el => {{
          renderKernelGraphIfNeeded(Number(el.dataset.kernelIndex));
        }});
      }});
    }});

    renderGraph(document.getElementById("pipeline-graph"),   REPORT.graphs.pipeline);
    renderGraph(document.getElementById("execution-graph"),  REPORT.graphs.execution);
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

    cpu_spans, cpu_compute_wall = _measure_cpu_schedule(schedule)

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
    # FIXME: These PCIe timings come from PCIeModel, not from a live FPGA DMA
    # measurement or host-side hardware timestamping path.
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
    # FIXME: FPGA wall time is derived from simulated cycles and synthesized Fmax,
    # not from direct timing on a programmed board.
    fpga_wall_s = (total_cycles / (fmax_mhz * 1e6)) if (fmax_mhz and fmax_mhz > 0) else None
    fpga_with_pcie_s = (fpga_wall_s + pcie_in_s + pcie_out_s) if fpga_wall_s is not None else None

    def _cycles_to_s(c):
        # FIXME: Per-phase FPGA times are derived from simulated cycles and
        # synthesized Fmax, not directly measured in hardware.
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
        schedule, pipeline_view, top,
        cycle_counts, cpu_spans, cpu_readback_wall,
        report["timing"], report["kernels"], fmax_mhz,
    )
    report["flamegraph"] = fg

    html_path = out_path / "index.html"
    html_path.write_text(_render_html(report, flamegraph_svg="",
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
