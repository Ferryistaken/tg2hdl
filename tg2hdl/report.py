from __future__ import annotations

import html
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tinygrad.engine.realize import run_schedule
from tinygrad.uop.ops import Ops
from tinygrad.viz.serve import uop_to_json

from compiler import compile_top_module, show_hardware, synthesis_stats
from compiler.top_module import simulate_top
from compiler.visualize import analyze_schedule


@dataclass
class BenchmarkArtifact:
    output_dir: str
    report_path: str
    tinygrad_device: str
    tinygrad_wall_s: float
    tg2hdl_wall_s: float
    tg2hdl_cycles: int
    correctness: bool


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


def _render_html(report: dict) -> str:
    report_json = json.dumps(report)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tg2hdl Benchmark</title>
  <script src="assets/d3js.org/d3.v7.min.js"></script>
  <script src="assets/dagrejs.github.io/project/dagre/latest/dagre.min.js"></script>
  <style>
    :root {{
      --bg: linear-gradient(135deg, #f5efe4 0%, #e8dfd0 45%, #d7e4ea 100%);
      --panel: rgba(255, 252, 247, 0.88);
      --ink: #182126;
      --muted: #4b5a61;
      --accent: #bb4d00;
      --accent2: #246a73;
      --ok: #1f7a43;
      --bad: #9f2d20;
      --edge: #8d8f91;
      --node: #fef7ec;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      min-height: 100vh;
    }}
    h1, h2, h3 {{
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      margin: 0;
    }}
    .shell {{
      width: min(1400px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      gap: 16px;
    }}
    .hero, .panel {{
      background: var(--panel);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(24,33,38,0.12);
      border-radius: 20px;
      box-shadow: 0 18px 50px rgba(24,33,38,0.08);
    }}
    .hero {{
      padding: 24px;
      display: grid;
      gap: 16px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .metric {{
      padding: 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.58);
      border: 1px solid rgba(24,33,38,0.08);
    }}
    .metric .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .metric .value {{
      font-size: 22px;
      font-weight: 700;
    }}
    .ok {{ color: var(--ok); }}
    .bad {{ color: var(--bad); }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .tab {{
      border: 0;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.65);
      color: var(--ink);
      cursor: pointer;
      font-weight: 600;
    }}
    .tab.active {{
      background: var(--ink);
      color: white;
    }}
    .panel {{
      padding: 18px;
      display: none;
      gap: 18px;
    }}
    .panel.active {{ display: grid; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      padding: 14px;
      border-radius: 16px;
      border: 1px solid rgba(24,33,38,0.08);
      background: rgba(255,255,255,0.62);
      overflow: hidden;
    }}
    .graph {{
      height: 520px;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      overflow: hidden;
    }}
    .graph svg {{
      width: 100%;
      height: 100%;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 13px;
      line-height: 1.45;
    }}
    .kernel-list {{
      display: grid;
      gap: 16px;
    }}
    .synth svg {{
      width: 100%;
      height: auto;
    }}
    .muted {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <h1>tg2hdl Benchmark</h1>
        <p class="muted">Tinygrad execution, tg2hdl compilation, graph scheduling, and FPGA-oriented synthesis in one report.</p>
      </div>
      <div class="metrics">
        <div class="metric"><div class="label">Correctness</div><div class="value {'ok' if report['summary']['correctness'] else 'bad'}">{'PASS' if report['summary']['correctness'] else 'FAIL'}</div></div>
        <div class="metric"><div class="label">Tinygrad Device</div><div class="value">{html.escape(report['summary']['tinygrad_device'])}</div></div>
        <div class="metric"><div class="label">Tinygrad Wall</div><div class="value">{report['summary']['tinygrad_wall_s']:.6f}s</div></div>
        <div class="metric"><div class="label">tg2hdl Wall</div><div class="value">{report['summary']['tg2hdl_wall_s']:.6f}s</div></div>
        <div class="metric"><div class="label">tg2hdl Cycles</div><div class="value">{report['summary']['tg2hdl_cycles']}</div></div>
        <div class="metric"><div class="label">Kernels</div><div class="value">{len(report['kernels'])}</div></div>
      </div>
      <div class="grid-2">
        <div class="card">
          <h2>What This Page Shows</h2>
          <pre>1. Tinygrad lowers the input tensor expression into one or more compute kernels.
2. tg2hdl compiles those kernels into KernelIR and Amaranth modules.
3. tg2hdl builds a top-level execution graph showing kernel dependencies and copies.
4. Tinygrad is executed as the reference implementation.
5. The tg2hdl top module is simulated in Amaranth and compared against that reference.
6. The report synthesizes both the full assembled system and each individual kernel for {FPGA_FAMILY} {FPGA_DEVICE.upper()}-{FPGA_PACKAGE}.</pre>
        </div>
        <div class="card">
          <h2>Provenance</h2>
          <pre>Tinygrad sections:
- Reference output and Tinygrad wall time
- Tinygrad kernel graphs in the "Tinygrad Kernels" tab

tg2hdl sections:
- Kernel DAG and Execution DAG
- KernelIR listings
- tg2hdl simulation wall time and cycle count

Amaranth / FPGA sections:
- Generated RTL schematic when Yosys + Graphviz are available
- Resource and Fmax estimates from Yosys + nextpnr for {FPGA_FAMILY} {FPGA_DEVICE.upper()}-{FPGA_PACKAGE}</pre>
        </div>
      </div>
      <div class="tabs">
        <button class="tab active" data-target="overview">Overview</button>
        <button class="tab" data-target="tinygrad">Tinygrad Kernels</button>
        <button class="tab" data-target="kernelir">KernelIR</button>
        <button class="tab" data-target="synth">Amaranth Synth</button>
      </div>
    </section>
    <section class="panel active" id="overview">
      <div class="grid-2">
        <div class="card">
          <h2>Kernel DAG</h2>
          <p class="muted">This graph is built by tg2hdl from the tinygrad schedule. Each node is a compiled kernel. Each edge means one kernel output buffer becomes another kernel input buffer.</p>
          <div class="graph" id="pipeline-graph"></div>
        </div>
        <div class="card">
          <h2>Execution DAG</h2>
          <p class="muted">This graph is also tg2hdl-specific. It shows the execution order after topological sorting plus the buffer-copy edges the top-level executor will actually drive.</p>
          <div class="graph" id="execution-graph"></div>
        </div>
      </div>
      <div class="grid-2">
        <div class="card">
          <h2>Reference Output</h2>
          <p class="muted">Produced by executing the captured tinygrad schedule on the selected tinygrad device.</p>
          <pre id="reference-output"></pre>
        </div>
        <div class="card">
          <h2>tg2hdl Output</h2>
          <p class="muted">Produced by simulating the generated top-level Amaranth module in software and reading back the final output buffer.</p>
          <pre id="sim-output"></pre>
        </div>
      </div>
      <div class="card">
        <h2>Reading Guide</h2>
        <pre>Read left to right:
- Tinygrad starts with a tensor expression and lowers it into kernels.
- tg2hdl interprets those kernels and emits KernelIR plus Amaranth hardware.
- The Kernel DAG shows logical data dependencies.
- The Execution DAG shows the actual top-level schedule tg2hdl simulates.
- A PASS means the tg2hdl simulated output matches tinygrad's result for this run.</pre>
      </div>
    </section>
    <section class="panel" id="tinygrad">
      <div class="kernel-list" id="tinygrad-kernels"></div>
    </section>
    <section class="panel" id="kernelir">
      <div class="kernel-list" id="kernelir-kernels"></div>
    </section>
    <section class="panel" id="synth">
      <div class="kernel-list" id="synth-kernels"></div>
    </section>
  </div>
  <script>
    const REPORT = {report_json};

    function renderGraph(container, graphData) {{
      const root = d3.select(container);
      root.selectAll("*").remove();
      const svg = root.append("svg");
      const inner = svg.append("g");
      const graph = new dagre.graphlib.Graph();
      graph.setGraph({{ rankdir: "LR", nodesep: 24, ranksep: 56, marginx: 24, marginy: 24 }});
      graph.setDefaultEdgeLabel(() => ({{}}));

      Object.entries(graphData).forEach(([id, node]) => {{
        graph.setNode(id, {{
          label: node.label,
          width: 220,
          height: 72,
          color: node.color || "#fef7ec",
        }});
      }});
      Object.entries(graphData).forEach(([id, node]) => {{
        (node.src || []).forEach(([edgeLabel, srcId]) => {{
          graph.setEdge(srcId, id, {{ label: edgeLabel }});
        }});
      }});

      dagre.layout(graph);
      const defs = svg.append("defs");
      defs.append("marker")
        .attr("id", "arrow")
        .attr("viewBox", "0 0 10 10")
        .attr("refX", 9)
        .attr("refY", 5)
        .attr("markerWidth", 7)
        .attr("markerHeight", 7)
        .attr("orient", "auto-start-reverse")
        .append("path")
        .attr("d", "M 0 0 L 10 5 L 0 10 z")
        .attr("fill", "#8d8f91");

      graph.edges().forEach(edge => {{
        const e = graph.edge(edge);
        const pts = e.points;
        const path = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis);
        inner.append("path")
          .attr("d", path(pts))
          .attr("fill", "none")
          .attr("stroke", "#8d8f91")
          .attr("stroke-width", 1.6)
          .attr("marker-end", "url(#arrow)");
        const mid = pts[Math.floor(pts.length / 2)];
        inner.append("text")
          .attr("x", mid.x)
          .attr("y", mid.y - 8)
          .attr("text-anchor", "middle")
          .attr("fill", "#4b5a61")
          .style("font-size", "12px")
          .text(e.label || "");
      }});

      graph.nodes().forEach(id => {{
        const n = graph.node(id);
        const g = inner.append("g").attr("transform", `translate(${{n.x - n.width/2}}, ${{n.y - n.height/2}})`);
        g.append("rect")
          .attr("rx", 16)
          .attr("ry", 16)
          .attr("width", n.width)
          .attr("height", n.height)
          .attr("fill", n.color)
          .attr("stroke", "#6f6e68")
          .attr("stroke-width", 1.2);
        const lines = n.label.split("\\n");
        lines.forEach((line, idx) => {{
          g.append("text")
            .attr("x", 16)
            .attr("y", 24 + idx * 18)
            .attr("fill", "#182126")
            .style("font-size", idx === 0 ? "14px" : "12px")
            .style("font-weight", idx === 0 ? 700 : 500)
            .text(line);
        }});
      }});

      const dims = graph.graph();
      svg.attr("viewBox", `0 0 ${{dims.width + 48}} ${{dims.height + 48}}`);
    }}

    function populateOverview() {{
      renderGraph(document.getElementById("pipeline-graph"), REPORT.graphs.pipeline);
      renderGraph(document.getElementById("execution-graph"), REPORT.graphs.execution);
      document.getElementById("reference-output").textContent = JSON.stringify(REPORT.summary.reference_output, null, 2);
      document.getElementById("sim-output").textContent = JSON.stringify(REPORT.summary.sim_output, null, 2);
    }}

    function populateTinygrad() {{
      const root = document.getElementById("tinygrad-kernels");
      REPORT.kernels.forEach((kernel, idx) => {{
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `<h2>Kernel ${{idx}}<span class="muted"> · source K${{kernel.source_index}}</span></h2><p class="muted">This graph comes directly from tinygrad's UOp graph for the scheduled kernel.</p><p class="muted">${{kernel.metadata.join(", ") || "no metadata"}}</p><div class="graph" id="tinygrad-graph-${{idx}}"></div>`;
        root.appendChild(card);
        renderGraph(card.querySelector(".graph"), kernel.tinygrad_graph);
      }});
    }}

    function populateKernelIR() {{
      const root = document.getElementById("kernelir-kernels");
      REPORT.kernels.forEach((kernel, idx) => {{
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `<h2>KernelIR ${{idx}}</h2><p class="muted">This textual IR is generated by tg2hdl after translating tinygrad UOps into the compiler's own typed kernel representation.</p><pre></pre>`;
        card.querySelector("pre").textContent = kernel.kernel_ir;
        root.appendChild(card);
      }});
    }}

    function populateSynth() {{
      const root = document.getElementById("synth-kernels");
      const topCard = document.createElement("div");
      topCard.className = "card synth";
      const topStats = REPORT.top_synth.synth_stats;
      topCard.innerHTML = `<h2>Full System</h2>
        <p class="muted">${{REPORT.top_synth.description}}</p>
        <p class="muted">This is the real whole-design synthesis target: one assembled TopModule, not a per-kernel estimate.</p>
        <div class="metrics">
          <div class="metric"><div class="label">FPGA Target</div><div class="value">${{topStats.fpga_family}} ${{topStats.fpga_device.toUpperCase()}}-${{topStats.fpga_package}}</div></div>
          <div class="metric"><div class="label">Fmax</div><div class="value">${{topStats.fmax_mhz === null ? "n/a" : topStats.fmax_mhz.toFixed(2) + " MHz"}}</div></div>
          <div class="metric"><div class="label">Synth Wall</div><div class="value">${{topStats.synth_wall_s === null ? "n/a" : topStats.synth_wall_s.toFixed(3) + "s"}}</div></div>
          <div class="metric"><div class="label">LUTs</div><div class="value">${{topStats.comb}}</div></div>
          <div class="metric"><div class="label">FFs</div><div class="value">${{topStats.ff}}</div></div>
          <div class="metric"><div class="label">BRAM</div><div class="value">${{topStats.dp16kd}}</div></div>
          <div class="metric"><div class="label">DSP</div><div class="value">${{topStats.mult18}}</div></div>
          <div class="metric"><div class="label">On-chip Bits</div><div class="value">${{topStats.mem_bits}}</div></div>
        </div>`;
      if (REPORT.top_synth.synth_svg) {{
        const wrap = document.createElement("div");
        wrap.innerHTML = REPORT.top_synth.synth_svg;
        topCard.appendChild(wrap);
      }} else {{
        const p = document.createElement("p");
        p.className = "muted";
        p.textContent = "Full-system schematic unavailable. Yosys or Graphviz is probably missing.";
        topCard.appendChild(p);
      }}
      root.appendChild(topCard);

      REPORT.kernels.forEach((kernel, idx) => {{
        const card = document.createElement("div");
        card.className = "card synth";
        const stats = kernel.synth_stats;
        card.innerHTML = `<h2>Synthesis ${{idx}}</h2>
          <p class="muted">This is a per-kernel synthesis view. It is useful for localizing cost, but it is not the full assembled design.</p>
          <div class="metrics">
            <div class="metric"><div class="label">FPGA Target</div><div class="value">${{stats.fpga_family}} ${{stats.fpga_device.toUpperCase()}}-${{stats.fpga_package}}</div></div>
            <div class="metric"><div class="label">Fmax</div><div class="value">${{stats.fmax_mhz === null ? "n/a" : stats.fmax_mhz.toFixed(2) + " MHz"}}</div></div>
            <div class="metric"><div class="label">Synth Wall</div><div class="value">${{stats.synth_wall_s === null ? "n/a" : stats.synth_wall_s.toFixed(3) + "s"}}</div></div>
            <div class="metric"><div class="label">LUTs</div><div class="value">${{stats.comb}}</div></div>
            <div class="metric"><div class="label">FFs</div><div class="value">${{stats.ff}}</div></div>
            <div class="metric"><div class="label">BRAM</div><div class="value">${{stats.dp16kd}}</div></div>
            <div class="metric"><div class="label">DSP</div><div class="value">${{stats.mult18}}</div></div>
            <div class="metric"><div class="label">On-chip Bits</div><div class="value">${{stats.mem_bits}}</div></div>
          </div>`;
        if (kernel.synth_svg) {{
          const wrap = document.createElement("div");
          wrap.innerHTML = kernel.synth_svg;
          card.appendChild(wrap);
        }} else {{
          const p = document.createElement("p");
          p.className = "muted";
          p.textContent = "Schematic unavailable. Yosys or Graphviz is probably missing.";
          card.appendChild(p);
        }}
        root.appendChild(card);
      }});
    }}

    document.querySelectorAll(".tab").forEach(btn => {{
      btn.addEventListener("click", () => {{
        document.querySelectorAll(".tab").forEach(el => el.classList.remove("active"));
        document.querySelectorAll(".panel").forEach(el => el.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(btn.dataset.target).classList.add("active");
      }});
    }});

    populateOverview();
    populateTinygrad();
    populateKernelIR();
    populateSynth();
  </script>
</body>
</html>
"""


def benchmark(tensor, *, out_dir: str = "tg2hdl_report") -> BenchmarkArtifact:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _copy_viz_assets(out_path)

    schedule = tensor.schedule()
    pipeline_view = analyze_schedule(schedule)
    top, _connections, kernel_specs = compile_top_module(schedule)
    input_data = _infer_input_data(schedule, pipeline_view, top)

    t0 = time.perf_counter()
    run_schedule(list(schedule), do_update_stats=False)
    compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
    ref_out = np.array(compute_items[-1].bufs[0].numpy(), copy=True)
    tinygrad_wall = time.perf_counter() - t0

    sim_out_raw, sim_cycles, sim_wall = simulate_top(top, input_data)
    sim_out = _decode_sim_output(sim_out_raw, tensor.dtype)

    if np.issubdtype(ref_out.dtype, np.floating):
        correctness = bool(np.allclose(ref_out.reshape(-1), sim_out.reshape(-1), rtol=1e-5, atol=1e-5))
    else:
        correctness = bool(np.array_equal(ref_out.reshape(-1), sim_out.reshape(-1)))

    report = {
        "summary": {
            "tinygrad_device": str(tensor.device),
            "tinygrad_wall_s": tinygrad_wall,
            "tg2hdl_wall_s": sim_wall,
            "tg2hdl_cycles": sim_cycles,
            "correctness": correctness,
            "reference_output": ref_out.reshape(-1).tolist(),
            "sim_output": sim_out.reshape(-1).tolist(),
        },
        "graphs": _graph_payload(pipeline_view),
        "top_synth": _top_payload(top, out_path / "synth"),
        "kernels": _kernel_payload(schedule, pipeline_view, kernel_specs, out_path / "synth"),
    }

    html_path = out_path / "index.html"
    html_path.write_text(_render_html(report))
    return BenchmarkArtifact(
        output_dir=str(out_path),
        report_path=str(html_path),
        tinygrad_device=str(tensor.device),
        tinygrad_wall_s=tinygrad_wall,
        tg2hdl_wall_s=sim_wall,
        tg2hdl_cycles=sim_cycles,
        correctness=correctness,
    )
