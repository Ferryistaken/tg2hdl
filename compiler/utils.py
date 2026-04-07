"""Utilities for inspecting tinygrad UOps and generated hardware."""

from typing import List, Optional
from textwrap import shorten


def format_uops(uops: List["UOp"], full_width: bool = False) -> str:
    """Format a linearized UOp list as a readable table.

    Each row: index, op name, dtype, arg, sources (by index).
    If full_width is True, columns expand to fit content without truncation.
    """
    from tinygrad import UOp

    # ---- 1. Collect raw string data first ----
    rows = []
    idx = {id(u): i for i, u in enumerate(uops)}

    # Header row
    rows.append({
        "idx": "#",
        "op": "op",
        "dtype": "dtype",
        "arg": "arg",
        "src": "src",
    })

    for i, u in enumerate(uops):
        src_ids = "[" + ", ".join(f"%{idx[id(s)]}" for s in u.src) + "]"
        rows.append({
            "idx": str(i),
            "op": str(u.op),
            "dtype": str(u.dtype),
            "arg": repr(u.arg),
            "src": src_ids,
        })

    # ---- 2. Determine column widths ----
    if full_width:
        # Use max length of actual content for each column
        W_IDX   = max(len(r["idx"])   for r in rows)
        W_OP    = max(len(r["op"])    for r in rows)
        W_DTYPE = max(len(r["dtype"]) for r in rows)
        W_ARG   = max(len(r["arg"])   for r in rows)
        W_SRC   = max(len(r["src"])   for r in rows)
        truncate = False
    else:
        # Fixed widths with truncation
        W_IDX   = 3
        W_OP    = 18
        W_DTYPE = 20
        W_ARG   = 20
        W_SRC   = 10
        truncate = True

    def fmt_cell(val: str, width: int, align_left: bool = True) -> str:
        if truncate:
            # Reserve space for ellipsis if we truncate
            val = shorten(val, width=width, placeholder="…")
        if align_left:
            return f"{val:<{width}}"
        else:
            return f"{val:>{width}}"

    # ---- 3. Build header and rows ----
    header = rows[0]
    header_line = (
        f"{fmt_cell(header['idx'],   W_IDX,   align_left=False)}  "
        f"{fmt_cell(header['op'],    W_OP)}  "
        f"{fmt_cell(header['dtype'], W_DTYPE)}  "
        f"{fmt_cell(header['arg'],   W_ARG)}  "
        f"{fmt_cell(header['src'],   W_SRC)}"
    )
    lines = [header_line, "-" * len(header_line)]

    for r in rows[1:]:
        line = (
            f"{fmt_cell(r['idx'],   W_IDX,   align_left=False)}  "
            f"{fmt_cell(r['op'],    W_OP)}  "
            f"{fmt_cell(r['dtype'], W_DTYPE)}  "
            f"{fmt_cell(r['arg'],   W_ARG)}  "
            f"{fmt_cell(r['src'],   W_SRC)}"
        )
        lines.append(line)
    return "\n".join(lines)


def pretty_print_uops(uops: List["UOp"], full_width: bool = False) -> None:
    """Print a linearized UOp list in a readable table format."""
    print(format_uops(uops, full_width=full_width))


# ---------------------------------------------------------------------------
# show_hardware — Yosys schematic generation
# ---------------------------------------------------------------------------

def show_hardware(kernel, out_dir: str, *,
                  stage: str = "opt",
                  fmt: str = "svg") -> Optional[str]:
    """Generate a Yosys schematic of a CompiledKernel.

    Parameters
    ----------
    kernel : CompiledKernel
        The compiled kernel module.
    out_dir : str
        Directory to write the output file.
    stage : str
        Level of Yosys lowering before drawing:
          "rtl"     — raw RTLIL structure (closest to source)
          "opt"     — after proc + opt + clean (recommended default)
          "generic" — after technology mapping to generic cells
          "mapped"  — after full ECP5 synthesis (very detailed)
    fmt : str
        Output format: "svg" (default), "pdf", or "png".

    Returns
    -------
    str or None
        Path to the generated file, or None if yosys/dot not available.
    """
    import os
    import shutil
    import subprocess
    import tempfile
    from amaranth.back import rtlil

    if not shutil.which("yosys") or not shutil.which("dot"):
        return None

    os.makedirs(out_dir, exist_ok=True)

    if stage == "rtl":
        passes = "hierarchy -top top; opt_clean; clean;"
    elif stage == "opt":
        passes = "hierarchy -top top; proc;; opt;; clean;"
    elif stage == "generic":
        passes = "hierarchy -top top; proc;; opt;; clean; techmap;; opt;; clean;"
    elif stage == "mapped":
        passes = "hierarchy -top top; synth_ecp5;; clean;"
    else:
        raise ValueError(f"Unknown stage {stage!r}")

    il = rtlil.convert(kernel, ports=[kernel.start, kernel.done, kernel.busy])
    prefix = os.path.join(out_dir, f"hardware_{stage}")
    out_path = f"{prefix}.{fmt}"

    with tempfile.TemporaryDirectory() as d:
        il_path = os.path.join(d, "top.il")
        with open(il_path, "w") as f:
            f.write(il)

        script = (
            f"read_rtlil {il_path}; "
            f"{passes} "
            f"show -format {fmt} -prefix {prefix} top"
        )
        r = subprocess.run(
            ["yosys", "-q", "-p", script],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            return None

    return out_path if os.path.exists(out_path) else None


# ---------------------------------------------------------------------------
# synthesis_stats — Yosys + nextpnr resource/timing analysis
# ---------------------------------------------------------------------------

def synthesis_stats(kernel, device: str = "45k", package: str = "CABGA381"):
    """Run Yosys + nextpnr-ecp5 and return resource/timing data.

    Returns a dict with:
      mem_bits    -- total on-chip storage in bits (from buf_infos)
      fp32_units  -- FP32 submodule count (from RTLIL)
      fmax_mhz    -- achieved Fmax in MHz (float), or None if unavailable
      comb        -- TRELLIS_COMB cells used (LUT equivalent)
      ff          -- TRELLIS_FF flip-flops used
      dp16kd      -- DP16KD block RAM tiles used
      mult18      -- MULT18X18D DSP multiplier tiles used
      from_synth  -- True when Yosys+nextpnr ran successfully

    Falls back gracefully when Yosys or nextpnr-ecp5 is not on PATH.
    """
    import os
    import shutil
    import subprocess
    import tempfile
    import json
    from amaranth.back import rtlil

    mem_bits = sum(b["depth"] * b["elem_width"] for b in kernel.buf_infos)
    fp32_units = _rtlil_fp32_units(kernel)

    base = dict(
        mem_bits=mem_bits,
        fp32_units=fp32_units,
        fmax_mhz=None,
        comb=0,
        ff=0,
        dp16kd=0,
        mult18=0,
        from_synth=False,
    )

    if not shutil.which("yosys") or not shutil.which("nextpnr-ecp5"):
        return base

    il = rtlil.convert(kernel, ports=[kernel.start, kernel.done, kernel.busy])

    with tempfile.TemporaryDirectory() as d:
        il_path = f"{d}/top.il"
        json_path = f"{d}/top.json"
        report_path = f"{d}/report.json"

        with open(il_path, "w") as f:
            f.write(il)

        r = subprocess.run(
            ["yosys", "-q", "-p", f"read_rtlil {il_path}; synth_ecp5 -json {json_path}"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            return base

        r2 = subprocess.run(
            ["nextpnr-ecp5", f"--{device}", "--package", package,
             "--json", json_path, "--report", report_path,
             "--timing-allow-fail", "--quiet"],
            capture_output=True, text=True, timeout=300,
        )
        if r2.returncode != 0:
            return base

        try:
            with open(report_path) as f:
                rep = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return base

    fmax_mhz = None
    for clk_data in rep.get("fmax", {}).values():
        achieved = clk_data.get("achieved")
        if achieved and (fmax_mhz is None or achieved < fmax_mhz):
            fmax_mhz = float(achieved)

    util = rep.get("utilization", {})
    return dict(
        mem_bits=mem_bits,
        fp32_units=fp32_units,
        fmax_mhz=fmax_mhz,
        comb=util.get("TRELLIS_COMB",  {}).get("used", 0),
        ff=util.get("TRELLIS_FF",     {}).get("used", 0),
        dp16kd=util.get("DP16KD",     {}).get("used", 0),
        mult18=util.get("MULT18X18D", {}).get("used", 0),
        from_synth=True,
    )


def _rtlil_fp32_units(kernel) -> int:
    """Count FP32Add/Mul/Cmp submodule instances from RTLIL (no Yosys needed)."""
    from amaranth.back import rtlil
    from collections import Counter
    il = rtlil.convert(kernel, ports=[kernel.start, kernel.done, kernel.busy])
    cell_counts = Counter()
    for line in il.split('\n'):
        s = line.strip()
        if s.startswith('cell '):
            cell_counts[s.split()[1]] += 1
    return sum(v for k, v in cell_counts.items() if k.startswith(r'\top.fp'))
