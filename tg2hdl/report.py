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
      --bg: #f4f6f8;
      --bg2: #edf1f4;
      --panel: #ffffff;
      --panel2: #fbfcfd;
      --line: #d7dde3;
      --line2: #aab6c2;
      --ink: #182126;
      --muted: #5b6975;
      --accent: #d88412;
      --accent2: #2a7db8;
      --ok: #1f7a43;
      --bad: #b13a2e;
      --edge: #7d8790;
      --node: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(180deg, rgba(216,132,18,0.06), transparent 240px),
        linear-gradient(90deg, rgba(24,33,38,0.03) 1px, transparent 1px),
        linear-gradient(0deg, rgba(24,33,38,0.02) 1px, transparent 1px),
        var(--bg);
      background-size: auto, 24px 24px, 24px 24px, auto;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      min-height: 100vh;
    }}
    h1, h2, h3 {{
      margin: 0;
      letter-spacing: 0.01em;
      font-weight: 650;
    }}
    .shell {{
      width: min(1480px, calc(100vw - 32px));
      margin: 16px auto 32px;
      display: grid;
      gap: 12px;
    }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
    }}
    .hero {{
      padding: 18px;
      display: grid;
      gap: 12px;
      background:
        linear-gradient(180deg, rgba(216,132,18,0.05), transparent 60%),
        var(--panel);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
    }}
    .metric {{
      padding: 12px;
      border-radius: 6px;
      background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(244,246,248,0.9));
      border: 1px solid var(--line);
      min-height: 84px;
    }}
    .metric .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .metric .value {{
      font-size: 24px;
      font-weight: 700;
      line-height: 1.1;
    }}
    .ok {{ color: var(--ok); }}
    .bad {{ color: var(--bad); }}
    .summary-grid {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 12px;
    }}
    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .summary-table td {{
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    .summary-table td:first-child {{
      width: 220px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 11px;
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}
    .tab {{
      border: 1px solid var(--line2);
      padding: 9px 13px;
      border-radius: 4px;
      background: var(--panel2);
      color: var(--ink);
      cursor: pointer;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 12px;
    }}
    .tab.active {{
      background: var(--accent);
      color: #ffffff;
      border-color: var(--accent);
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
      border-radius: 6px;
      border: 1px solid var(--line);
      background: var(--panel2);
      overflow: hidden;
    }}
    .graph {{
      height: 520px;
      border-radius: 6px;
      background:
        radial-gradient(circle at top left, rgba(42,125,184,0.05), transparent 35%),
        #ffffff;
      border: 1px solid var(--line);
      overflow: hidden;
      position: relative;
    }}
    .graph svg {{
      width: 100%;
      height: 100%;
    }}
    .graph-toolbar {{
      position: absolute;
      top: 8px;
      right: 8px;
      display: flex;
      gap: 6px;
      z-index: 5;
    }}
    .graph-btn {{
      border: 1px solid var(--line2);
      background: rgba(255,255,255,0.94);
      color: var(--ink);
      border-radius: 4px;
      padding: 4px 7px;
      font-size: 11px;
      cursor: pointer;
    }}
    .graph-detail {{
      position: absolute;
      left: 10px;
      bottom: 10px;
      max-width: 340px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.97);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 12px;
      color: var(--ink);
      box-shadow: 0 8px 24px rgba(0,0,0,0.08);
      pointer-events: none;
      opacity: 0;
      transition: opacity 120ms ease;
      white-space: pre-wrap;
    }}
    .graph-detail.visible {{
      opacity: 1;
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
    .section-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .badge {{
      border: 1px solid var(--line2);
      border-radius: 4px;
      padding: 3px 8px;
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .help {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      border-radius: 999px;
      border: 1px solid var(--line2);
      color: var(--accent);
      font-size: 11px;
      font-weight: 700;
      cursor: help;
      position: relative;
      flex: 0 0 auto;
    }}
    .help:hover::after {{
      content: attr(data-help);
      position: absolute;
      left: 22px;
      top: -8px;
      width: 260px;
      padding: 10px 12px;
      border-radius: 6px;
      border: 1px solid var(--line2);
      background: #101317;
      color: #eef3f6;
      text-transform: none;
      letter-spacing: 0;
      font-size: 12px;
      line-height: 1.4;
      white-space: normal;
      z-index: 20;
      box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    }}
    .methodology {{
      font-size: 13px;
      line-height: 1.55;
    }}
    .methodology p {{
      margin: 0 0 10px 0;
    }}
    .methodology code {{
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
    }}
    @media (max-width: 980px) {{
      .summary-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="section-head">
        <h1>tg2hdl Benchmark</h1>
        <div class="badge">Hardware Compilation Report</div>
      </div>
      <p class="muted">Tinygrad reference execution, tg2hdl lowering, top-level scheduling, and FPGA-oriented synthesis collected into a single engineering report.</p>
      <div class="summary-grid">
        <div class="card">
          <div class="section-head">
            <h2>Summary</h2>
            <div class="badge">At a Glance</div>
          </div>
          <table class="summary-table">
            <tr><td>Status</td><td class="{'ok' if report['summary']['correctness'] else 'bad'}"><strong>{'PASS' if report['summary']['correctness'] else 'FAIL'}</strong></td></tr>
            <tr><td>Tinygrad Device</td><td>{html.escape(report['summary']['tinygrad_device'])}</td></tr>
            <tr><td>Kernel Count</td><td>{len(report['kernels'])}</td></tr>
            <tr><td>FPGA Target</td><td>{FPGA_FAMILY} {FPGA_DEVICE.upper()}-{FPGA_PACKAGE}</td></tr>
            <tr><td>Report Scope</td><td>Reference execution, compiler graphs, KernelIR, full-system synthesis, per-kernel synthesis</td></tr>
          </table>
        </div>
        <div class="card">
          <div class="section-head">
            <h2>Primary Metrics</h2>
            <div class="badge">Measured</div>
          </div>
          <div class="metrics">
            <div class="metric"><div class="label">Correctness <span class="help" data-help="PASS means the final tg2hdl simulated output buffer exactly matched the tinygrad reference output for this run.">?</span></div><div class="value {'ok' if report['summary']['correctness'] else 'bad'}">{'PASS' if report['summary']['correctness'] else 'FAIL'}</div></div>
            <div class="metric"><div class="label">Tinygrad Wall <span class="help" data-help="Wall-clock time spent executing the captured tinygrad schedule on the selected tinygrad device, measured in Python around schedule execution.">?</span></div><div class="value">{report['summary']['tinygrad_wall_s']:.6f}s</div></div>
            <div class="metric"><div class="label">tg2hdl Wall <span class="help" data-help="Wall-clock time spent simulating the generated tg2hdl TopModule in Amaranth. This is software simulation time, not predicted FPGA runtime.">?</span></div><div class="value">{report['summary']['tg2hdl_wall_s']:.6f}s</div></div>
            <div class="metric"><div class="label">tg2hdl Cycles <span class="help" data-help="Clock cycles observed between start and done in the Amaranth TopModule simulation. This is the hardware-style runtime metric for the generated design.">?</span></div><div class="value">{report['summary']['tg2hdl_cycles']}</div></div>
            <div class="metric"><div class="label">Kernel Count <span class="help" data-help="Number of tinygrad compute kernels present in the scheduled program after lowering and fusion decisions.">?</span></div><div class="value">{len(report['kernels'])}</div></div>
            <div class="metric"><div class="label">Full-System Fmax <span class="help" data-help="Estimated maximum clock frequency reported by nextpnr for the full assembled tg2hdl TopModule targeting the selected FPGA.">?</span></div><div class="value">{_format_mhz(report['top_synth']['synth_stats']['fmax_mhz'])}</div></div>
          </div>
        </div>
      </div>
      <div class="tabs">
        <button class="tab active" data-target="summary">Summary</button>
        <button class="tab" data-target="overview">Graphs</button>
        <button class="tab" data-target="tinygrad">Tinygrad Kernels</button>
        <button class="tab" data-target="kernelir">KernelIR</button>
        <button class="tab" data-target="synth">Amaranth Synth</button>
        <button class="tab" data-target="methodology">Methodology</button>
      </div>
    </section>
    <section class="panel active" id="summary">
      <div class="grid-2">
        <div class="card">
          <div class="section-head">
            <h2>Outputs</h2>
            <div class="badge">Comparison</div>
          </div>
          <p class="muted">Reference output comes from tinygrad schedule execution. tg2hdl output comes from Amaranth simulation of the generated TopModule.</p>
          <div class="grid-2">
            <div>
              <div class="section-head"><h3>Reference</h3><div class="badge">tinygrad</div></div>
              <pre id="reference-output"></pre>
            </div>
            <div>
              <div class="section-head"><h3>tg2hdl</h3><div class="badge">simulation</div></div>
              <pre id="sim-output"></pre>
            </div>
          </div>
        </div>
        <div class="card">
          <div class="section-head">
            <h2>System Summary</h2>
            <div class="badge">Whole Design</div>
          </div>
          <div class="metrics">
            <div class="metric"><div class="label">FPGA Target <span class="help" data-help="Synthesis target used for both the full-system and per-kernel FPGA estimates.">?</span></div><div class="value" style="font-size:16px;">{FPGA_FAMILY} {FPGA_DEVICE.upper()}-{FPGA_PACKAGE}</div></div>
            <div class="metric"><div class="label">Synth Wall <span class="help" data-help="Wall-clock time spent running Yosys plus nextpnr for the full TopModule, when available.">?</span></div><div class="value">{_format_seconds(report['top_synth']['synth_stats']['synth_wall_s'])}</div></div>
            <div class="metric"><div class="label">LUTs <span class="help" data-help="TRELLIS_COMB cells used by the full assembled design according to nextpnr utilization output.">?</span></div><div class="value">{report['top_synth']['synth_stats']['comb']}</div></div>
            <div class="metric"><div class="label">FFs <span class="help" data-help="TRELLIS_FF cells used by the full assembled design according to nextpnr utilization output.">?</span></div><div class="value">{report['top_synth']['synth_stats']['ff']}</div></div>
            <div class="metric"><div class="label">BRAM <span class="help" data-help="DP16KD block RAM tiles used by the full assembled design according to nextpnr utilization output.">?</span></div><div class="value">{report['top_synth']['synth_stats']['dp16kd']}</div></div>
            <div class="metric"><div class="label">DSP <span class="help" data-help="MULT18X18D DSP multiplier tiles used by the full assembled design according to nextpnr utilization output.">?</span></div><div class="value">{report['top_synth']['synth_stats']['mult18']}</div></div>
          </div>
        </div>
      </div>
    </section>
    <section class="panel" id="overview">
      <div class="grid-2">
        <div class="card">
          <div class="section-head">
            <h2>Kernel DAG</h2>
            <div class="badge">tg2hdl</div>
          </div>
          <p class="muted">Logical kernel dependency graph reconstructed by tg2hdl from the tinygrad schedule.</p>
          <div class="graph" id="pipeline-graph"></div>
        </div>
        <div class="card">
          <div class="section-head">
            <h2>Execution DAG</h2>
            <div class="badge">tg2hdl</div>
          </div>
          <p class="muted">Top-level execution graph after tg2hdl ordering and copy-group construction.</p>
          <div class="graph" id="execution-graph"></div>
        </div>
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
    <section class="panel" id="methodology">
      <div class="card methodology">
        <div class="section-head">
          <h2>Methodology</h2>
          <div class="badge">Explanations</div>
        </div>
        <p><strong>Tinygrad reference:</strong> the input tensor expression is scheduled by tinygrad and executed through <code>run_schedule(...)</code>. The final realized output buffer is used as the reference result.</p>
        <p><strong>Kernel graphs:</strong> the “Tinygrad Kernels” tab comes directly from tinygrad’s UOp graph for each scheduled compute kernel.</p>
        <p><strong>Kernel DAG and Execution DAG:</strong> these are tg2hdl-generated views. The Kernel DAG shows logical producer/consumer dependencies. The Execution DAG shows the actual topological execution order and explicit copy edges the tg2hdl top-level executor will drive.</p>
        <p><strong>tg2hdl wall time:</strong> this is Python wall-clock time for Amaranth simulation of the generated <code>TopModule</code>. It is not intended as an estimate of FPGA runtime.</p>
        <p><strong>tg2hdl cycles:</strong> this is the cycle count observed from <code>start</code> to <code>done</code> in the Amaranth simulation. This is the closest report metric to generated hardware runtime.</p>
        <p><strong>Synth wall / Fmax / utilization:</strong> these come from Yosys plus nextpnr targeting <code>{FPGA_FAMILY} {FPGA_DEVICE.upper()}-{FPGA_PACKAGE}</code>. If these tools are not present, the report still renders but synthesis-specific values appear as unavailable.</p>
        <p><strong>Full System vs Per-Kernel synthesis:</strong> the top synth block represents the assembled design with all kernels and control logic. The per-kernel blocks are local cost views and do not include full-system orchestration overhead.</p>
      </div>
    </section>
  </div>
  <script>
    const REPORT = {report_json};

    function renderGraph(container, graphData) {{
      const root = d3.select(container);
      root.selectAll("*").remove();
      const toolbar = root.append("div").attr("class", "graph-toolbar");
      const detail = root.append("div").attr("class", "graph-detail").text("");
      const svg = root.append("svg");
      const viewport = svg.append("g");
      const inner = viewport.append("g");
      const graph = new dagre.graphlib.Graph();
      graph.setGraph({{ rankdir: "LR", nodesep: 24, ranksep: 56, marginx: 24, marginy: 24 }});
      graph.setDefaultEdgeLabel(() => ({{}}));

      Object.entries(graphData).forEach(([id, node]) => {{
        graph.setNode(id, {{
          label: node.label,
          width: 220,
          height: 72,
          color: node.color || "#20272d",
        }});
      }});
      Object.entries(graphData).forEach(([id, node]) => {{
        (node.src || []).forEach(([edgeLabel, srcId]) => {{
          graph.setEdge(srcId, id, {{ label: edgeLabel }});
        }});
      }});

      dagre.layout(graph);
      const zoom = d3.zoom().scaleExtent([0.35, 3]).on("zoom", (event) => {{
        viewport.attr("transform", event.transform);
      }});
      svg.call(zoom);

      function fitGraph() {{
        const dims = graph.graph();
        const width = container.clientWidth || 800;
        const height = container.clientHeight || 520;
        const scale = Math.min(width / Math.max(dims.width + 48, 1), height / Math.max(dims.height + 48, 1), 1);
        const tx = (width - dims.width * scale) / 2 + 24;
        const ty = (height - dims.height * scale) / 2 + 24;
        svg.transition().duration(250).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
      }}

      toolbar.append("button").attr("class", "graph-btn").text("Fit").on("click", fitGraph);
      toolbar.append("button").attr("class", "graph-btn").text("+").on("click", () => svg.transition().duration(180).call(zoom.scaleBy, 1.2));
      toolbar.append("button").attr("class", "graph-btn").text("-").on("click", () => svg.transition().duration(180).call(zoom.scaleBy, 0.85));

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
        .attr("fill", "#86929a");

      graph.edges().forEach(edge => {{
        const e = graph.edge(edge);
        const pts = e.points;
        const path = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis);
        inner.append("path")
          .attr("d", path(pts))
          .attr("fill", "none")
          .attr("stroke", "#7d8790")
          .attr("stroke-width", 1.6)
          .attr("marker-end", "url(#arrow)");
        const mid = pts[Math.floor(pts.length / 2)];
        inner.append("text")
          .attr("x", mid.x)
          .attr("y", mid.y - 8)
          .attr("text-anchor", "middle")
          .attr("fill", "#5b6975")
          .style("font-size", "12px")
          .text(e.label || "");
      }});

      graph.nodes().forEach(id => {{
        const n = graph.node(id);
        const g = inner.append("g").attr("transform", `translate(${{n.x - n.width/2}}, ${{n.y - n.height/2}})`);
        const rect = g.append("rect")
          .attr("rx", 16)
          .attr("ry", 16)
          .attr("width", n.width)
          .attr("height", n.height)
          .attr("fill", n.color)
          .attr("stroke", "#8b97a1")
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
        g.style("cursor", "pointer")
          .on("mouseenter", () => {{
            rect.attr("stroke", "#2a7db8").attr("stroke-width", 2);
            detail.classed("visible", true).text(n.label);
          }})
          .on("mouseleave", () => {{
            rect.attr("stroke", "#8b97a1").attr("stroke-width", 1.2);
            detail.classed("visible", false);
          }});
      }});

      const dims = graph.graph();
      svg.attr("viewBox", `0 0 ${{dims.width + 48}} ${{dims.height + 48}}`);
      fitGraph();
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
        card.innerHTML = `<div class="section-head"><h2>Kernel ${{idx}}</h2><div class="badge">tinygrad source K${{kernel.source_index}}</div></div><p class="muted">This graph comes directly from tinygrad's UOp graph for the scheduled kernel.</p><p class="muted">${{kernel.metadata.join(", ") || "no metadata"}}</p><div class="graph" id="tinygrad-graph-${{idx}}"></div>`;
        root.appendChild(card);
        renderGraph(card.querySelector(".graph"), kernel.tinygrad_graph);
      }});
    }}

    function populateKernelIR() {{
      const root = document.getElementById("kernelir-kernels");
      REPORT.kernels.forEach((kernel, idx) => {{
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `<div class="section-head"><h2>KernelIR ${{idx}}</h2><div class="badge">tg2hdl</div></div><p class="muted">This textual IR is generated by tg2hdl after translating tinygrad UOps into the compiler's own typed kernel representation.</p><pre></pre>`;
        card.querySelector("pre").textContent = kernel.kernel_ir;
        root.appendChild(card);
      }});
    }}

    function populateSynth() {{
      const root = document.getElementById("synth-kernels");
      const topCard = document.createElement("div");
      topCard.className = "card synth";
      const topStats = REPORT.top_synth.synth_stats;
      topCard.innerHTML = `<div class="section-head"><h2>Full System</h2><div class="badge">assembled design</div></div>
        <p class="muted">${{REPORT.top_synth.description}}</p>
        <p class="muted">This is the real whole-design synthesis target: one assembled TopModule, not a per-kernel estimate.</p>
        <div class="metrics">
          <div class="metric"><div class="label">FPGA Target <span class="help" data-help="Synthesis target used for the full assembled TopModule.">?</span></div><div class="value" style="font-size:16px;">${{topStats.fpga_family}} ${{topStats.fpga_device.toUpperCase()}}-${{topStats.fpga_package}}</div></div>
          <div class="metric"><div class="label">Fmax <span class="help" data-help="Estimated maximum clock frequency reported by nextpnr for the full assembled design.">?</span></div><div class="value">${{topStats.fmax_mhz === null ? "n/a" : topStats.fmax_mhz.toFixed(2) + " MHz"}}</div></div>
          <div class="metric"><div class="label">Synth Wall <span class="help" data-help="Wall-clock time spent running Yosys plus nextpnr for the full assembled TopModule.">?</span></div><div class="value">${{topStats.synth_wall_s === null ? "n/a" : topStats.synth_wall_s.toFixed(3) + "s"}}</div></div>
          <div class="metric"><div class="label">LUTs <span class="help" data-help="TRELLIS_COMB cells used by the full design according to nextpnr utilization output.">?</span></div><div class="value">${{topStats.comb}}</div></div>
          <div class="metric"><div class="label">FFs <span class="help" data-help="TRELLIS_FF cells used by the full design according to nextpnr utilization output.">?</span></div><div class="value">${{topStats.ff}}</div></div>
          <div class="metric"><div class="label">BRAM <span class="help" data-help="DP16KD block RAM tiles used by the full design according to nextpnr utilization output.">?</span></div><div class="value">${{topStats.dp16kd}}</div></div>
          <div class="metric"><div class="label">DSP <span class="help" data-help="MULT18X18D DSP multiplier tiles used by the full design according to nextpnr utilization output.">?</span></div><div class="value">${{topStats.mult18}}</div></div>
          <div class="metric"><div class="label">On-chip Bits <span class="help" data-help="Total buffer storage bits across the compiled design, aggregated from compiled kernel buffer definitions.">?</span></div><div class="value">${{topStats.mem_bits}}</div></div>
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
        card.innerHTML = `<div class="section-head"><h2>Synthesis ${{idx}}</h2><div class="badge">per-kernel</div></div>
          <p class="muted">This is a per-kernel synthesis view. It is useful for localizing cost, but it is not the full assembled design.</p>
          <div class="metrics">
            <div class="metric"><div class="label">FPGA Target <span class="help" data-help="Synthesis target used for this kernel-local estimate.">?</span></div><div class="value" style="font-size:16px;">${{stats.fpga_family}} ${{stats.fpga_device.toUpperCase()}}-${{stats.fpga_package}}</div></div>
            <div class="metric"><div class="label">Fmax <span class="help" data-help="Estimated maximum clock frequency reported by nextpnr for this compiled kernel alone.">?</span></div><div class="value">${{stats.fmax_mhz === null ? "n/a" : stats.fmax_mhz.toFixed(2) + " MHz"}}</div></div>
            <div class="metric"><div class="label">Synth Wall <span class="help" data-help="Wall-clock time spent running Yosys plus nextpnr for this compiled kernel alone.">?</span></div><div class="value">${{stats.synth_wall_s === null ? "n/a" : stats.synth_wall_s.toFixed(3) + "s"}}</div></div>
            <div class="metric"><div class="label">LUTs <span class="help" data-help="TRELLIS_COMB cells used by this kernel according to nextpnr utilization output.">?</span></div><div class="value">${{stats.comb}}</div></div>
            <div class="metric"><div class="label">FFs <span class="help" data-help="TRELLIS_FF cells used by this kernel according to nextpnr utilization output.">?</span></div><div class="value">${{stats.ff}}</div></div>
            <div class="metric"><div class="label">BRAM <span class="help" data-help="DP16KD block RAM tiles used by this kernel according to nextpnr utilization output.">?</span></div><div class="value">${{stats.dp16kd}}</div></div>
            <div class="metric"><div class="label">DSP <span class="help" data-help="MULT18X18D DSP tiles used by this kernel according to nextpnr utilization output.">?</span></div><div class="value">${{stats.mult18}}</div></div>
            <div class="metric"><div class="label">On-chip Bits <span class="help" data-help="Total buffer storage bits for this compiled kernel.">?</span></div><div class="value">${{stats.mem_bits}}</div></div>
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
