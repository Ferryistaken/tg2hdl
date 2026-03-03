"""Compare MNIST inference across four paths.

Path A: tinygrad float32 on CPU (reference).
Path G: tinygrad float32 on GPU (if available; CUDA / NV / HIP / METAL probed).
Path B: tinygrad UOps → HDL compiler → Amaranth simulation (float32, IEEE 754).
Path C: tinygrad UOps → HDL compiler → Amaranth simulation (INT8 quantized).

Both HDL paths use the generic compile_kernel pipeline. Float32 arithmetic
uses FP32Add / FP32Mul / FP32Cmp hardware modules — the same elaboratable
design that goes to synthesis.

Warning: full MNIST simulation (100k+ cycles) takes several minutes per path.
"""

import os
from contextlib import contextmanager

os.environ["DEBUG"] = "0"

import math
import time
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import NOOPT as tg_noopt
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import safe_load
from tinygrad.uop.ops import Ops

from compiler import HDLRenderer, compile_kernel, simulate_kernel
from utils import quantize_int8
from compiler.backend import _get_uops


@contextmanager
def _noopt_scope(value=1):
    old = tg_noopt.value
    tg_noopt.value = value
    try:
        yield
    finally:
        tg_noopt.value = old


# ---------------------------------------------------------------------------
# Hardware resource estimation
# ---------------------------------------------------------------------------

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


def _synthesis_stats(kernel, device="45k", package="CABGA381"):
    """Run Yosys + nextpnr-ecp5 and return real resource/timing data.

    Returns a dict with:
      fmax_mhz    -- achieved Fmax in MHz (float), or None if unavailable
      comb        -- TRELLIS_COMB cells used (LUT equivalent)
      ff          -- TRELLIS_FF flip-flops used
      dp16kd      -- DP16KD block RAM tiles used
      mult18      -- MULT18X18D DSP multiplier tiles used
      mem_bits    -- total on-chip storage in bits (exact, from buf_infos)
      fp32_units  -- FP32 submodule count (from RTLIL)
      from_synth  -- True when Yosys+nextpnr ran successfully

    Falls back gracefully when Yosys or nextpnr-ecp5 is not on PATH.
    """
    import shutil, subprocess, tempfile, json
    from amaranth.back import rtlil

    mem_bits   = sum(b["depth"] * b["elem_width"] for b in kernel.buf_infos)
    fp32_units = _rtlil_fp32_units(kernel)
    base = dict(mem_bits=mem_bits, fp32_units=fp32_units,
                fmax_mhz=None, comb=0, ff=0, dp16kd=0, mult18=0, from_synth=False)

    if not shutil.which("yosys") or not shutil.which("nextpnr-ecp5"):
        return base

    il = rtlil.convert(kernel, ports=[kernel.start, kernel.done, kernel.busy])

    with tempfile.TemporaryDirectory() as d:
        il_path     = f"{d}/top.il"
        json_path   = f"{d}/top.json"
        report_path = f"{d}/report.json"

        with open(il_path, "w") as f:
            f.write(il)

        # Yosys: synthesise for ECP5
        r = subprocess.run(
            ["yosys", "-q", "-p", f"read_rtlil {il_path}; synth_ecp5 -json {json_path}"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            return base

        # nextpnr-ecp5: place & route, emit timing report
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

    # Extract Fmax: take minimum achieved over all clocks (most conservative)
    fmax_mhz = None
    for clk_data in rep.get("fmax", {}).values():
        achieved = clk_data.get("achieved")
        if achieved and (fmax_mhz is None or achieved < fmax_mhz):
            fmax_mhz = float(achieved)

    util = rep.get("utilization", {})
    return dict(
        mem_bits   = mem_bits,
        fp32_units = fp32_units,
        fmax_mhz   = fmax_mhz,
        comb       = util.get("TRELLIS_COMB",  {}).get("used", 0),
        ff         = util.get("TRELLIS_FF",     {}).get("used", 0),
        dp16kd     = util.get("DP16KD",         {}).get("used", 0),
        mult18     = util.get("MULT18X18D",     {}).get("used", 0),
        from_synth = True,
    )


# Xilinx RAMB36 = 36 Kbits; one block covers up to 36,864 bits of storage.
_RAMB36_BITS = 36 * 1024


# ---------------------------------------------------------------------------
# GPU detection + inference
# ---------------------------------------------------------------------------

# Backends tried in priority order.  tinygrad will raise if the runtime is
# absent or the device is not present.
_GPU_BACKENDS = ['CUDA', 'NV', 'HIP', 'GPU', 'METAL']


def _detect_gpu():
    """Return the first working tinygrad GPU device string, or None."""
    for dev in _GPU_BACKENDS:
        try:
            t = Tensor([1.0, 2.0], device=dev)
            _ = t.numpy()   # force execution + synchronise
            return dev
        except Exception:
            continue
    return None


def _gpu_inference(x_float, w1, b1, w2, b2, device, warmup=3, runs=50):
    """Run MNIST forward pass on *device*, return (pred, avg_ms).

    Does *warmup* un-timed passes (kernel compilation / JIT tracing), then
    averages *runs* timed passes.  Each pass calls .numpy() to synchronise
    before stopping the clock.
    """
    x_t  = Tensor(x_float.reshape(1, 784), device=device)
    w1_t = Tensor(w1, device=device)
    b1_t = Tensor(b1, device=device)
    w2_t = Tensor(w2, device=device)
    b2_t = Tensor(b2, device=device)

    def _forward():
        h      = (x_t.linear(w1_t.T, b1_t)).relu()
        logits = h.linear(w2_t.T, b2_t)
        return logits.numpy().flatten()

    for _ in range(warmup):
        _forward()

    t0 = time.perf_counter()
    for _ in range(runs):
        out = _forward()
    avg_ms = (time.perf_counter() - t0) / runs * 1e3

    return int(out.argmax()), avg_ms


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

def _compile_kernels(sched):
    renderer = HDLRenderer()
    kernels = []
    with _noopt_scope(1):
        for si in sched:
            if si.ast.op != Ops.SINK:
                continue
            kernels.append(compile_kernel(_get_uops(si.ast, renderer)))
    return kernels


def _build_fp32_schedule():
    x  = Tensor.empty(1, 784, dtype=dtypes.float32)
    w1 = Tensor.empty(784, 128, dtype=dtypes.float32)
    b1 = Tensor.empty(1, 128, dtype=dtypes.float32)
    w2 = Tensor.empty(128, 10, dtype=dtypes.float32)
    b2 = Tensor.empty(1, 10, dtype=dtypes.float32)
    h      = ((x @ w1) + b1).relu()
    logits = (h @ w2) + b2
    return logits.schedule()


def _build_int8_schedule():
    x  = Tensor.empty(1, 784, dtype=dtypes.int8)
    w1 = Tensor.empty(784, 128, dtype=dtypes.int8)
    b1 = Tensor.empty(1, 128, dtype=dtypes.int32)
    w2 = Tensor.empty(128, 10, dtype=dtypes.int8)
    b2 = Tensor.empty(1, 10, dtype=dtypes.int32)
    h      = ((x @ w1).cast(dtypes.int32) + b1).relu()
    logits = (h.cast(dtypes.int8) @ w2).cast(dtypes.int32) + b2
    return logits.schedule()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Weights ---
    state = safe_load("mnist_weights.safetensors")
    w1 = state["l1.weight"].numpy()   # (128, 784) float32
    b1 = state["l1.bias"].numpy()     # (128,)     float32
    w2 = state["l2.weight"].numpy()   # (10, 128)  float32
    b2 = state["l2.bias"].numpy()     # (10,)      float32

    # --- Test image ---
    _, _, X_test, y_test = mnist()
    img_idx = 0
    x_float = X_test[img_idx].numpy().reshape(784).astype(np.float32) / 255.0
    label   = int(y_test[img_idx].numpy())

    print("=" * 60)
    print("MNIST Inference Comparison  (A=CPU · B=FP32-HDL · C=INT8-HDL)")
    print("=" * 60)
    print(f"Test image index: {img_idx}, true label: {label}")
    print()

    # ================================================================
    # Path A: tinygrad float32 CPU
    # ================================================================
    Tensor.training = False
    x_t  = Tensor(x_float.reshape(1, 784))
    w1_t, b1_t = Tensor(w1), Tensor(b1)
    w2_t, b2_t = Tensor(w2), Tensor(b2)

    t0 = time.perf_counter()
    h_cpu     = (x_t.linear(w1_t.T, b1_t)).relu()
    logits_cpu = h_cpu.linear(w2_t.T, b2_t)
    logits_cpu_np = logits_cpu.numpy().flatten()
    pred_cpu  = int(logits_cpu_np.argmax())
    t_cpu     = time.perf_counter() - t0

    mark = "\u2713" if pred_cpu == label else "\u2717"
    print(f"Path A \u2014 tinygrad float32 CPU:")
    print(f"  prediction: {pred_cpu} {mark}")
    print(f"  wall-clock: {t_cpu*1000:.2f} ms")
    print()

    # ================================================================
    # Path G: GPU (if available)
    # ================================================================
    print("Path G \u2014 tinygrad float32 GPU:", end=" ", flush=True)
    gpu_device = _detect_gpu()
    pred_gpu = gpu_ms = None
    if gpu_device is None:
        print("no GPU detected (tried: " + ", ".join(_GPU_BACKENDS) + ")")
    else:
        print(f"{gpu_device} detected, running...", end=" ", flush=True)
        try:
            pred_gpu, gpu_ms = _gpu_inference(x_float, w1, b1, w2, b2, gpu_device)
            mark = "\u2713" if pred_gpu == label else "\u2717"
            print(f"done")
            print(f"  prediction: {pred_gpu} {mark}")
            print(f"  avg of 50 runs: {gpu_ms:.3f} ms  (after 3 warmup passes)")
        except Exception as e:
            print(f"failed ({e})")
            gpu_device = None
    print()

    # ================================================================
    # Path B: float32 HDL simulation
    # ================================================================
    print("Path B \u2014 float32 HDL simulation (FP32Add / FP32Mul / FP32Cmp):")
    print("  Note: full MNIST simulation (~100k cycles) may take several minutes.")
    print()

    with _noopt_scope(1):
        kernels_fp32 = _compile_kernels(_build_fp32_schedule())
    print(f"  compiled {len(kernels_fp32)} kernels")
    print(f"  synthesising for ECP5 45F (Yosys + nextpnr-ecp5)...", end=" ", flush=True)
    stats_fp32 = [_synthesis_stats(k) for k in kernels_fp32]
    if stats_fp32[0]["from_synth"]:
        print("done")
    else:
        print("tools not found, using RTLIL estimates")

    # tinygrad layout: x(1,784) @ w(784,128) — so weights must be transposed
    # from safetensors (128,784) → (784,128)
    w1_fp32 = w1.T.astype(np.float32)   # (784, 128) float32
    w2_fp32 = w2.T.astype(np.float32)   # (128, 10)  float32

    print(f"  running kernel 0 (128\u00d7784 GEMV + bias + ReLU)...", end=" ", flush=True)
    out1_fp32, cyc0_fp32, wall0_fp32 = simulate_kernel(
        kernels_fp32[0],
        {1: x_float,         # (784,)    float32
         2: w1_fp32.flatten(), # (100352,) float32
         3: b1},              # (128,)    float32
    )
    print(f"done ({wall0_fp32:.1f}s)")

    # simulate_kernel returns int32 bit-patterns; reinterpret as float32
    hidden_fp32 = out1_fp32.astype(np.uint32).view(np.float32)

    print(f"  running kernel 1 (10\u00d7128 GEMV + bias)...", end=" ", flush=True)
    out2_fp32, cyc1_fp32, wall1_fp32 = simulate_kernel(
        kernels_fp32[1],
        {1: hidden_fp32,       # (128,)    float32 from kernel 0
         2: w2_fp32.flatten(), # (1280,)   float32
         3: b2},               # (10,)     float32
    )
    print(f"done ({wall1_fp32:.1f}s)")

    logits_fp32_hdl = out2_fp32.astype(np.uint32).view(np.float32)
    pred_fp32       = int(logits_fp32_hdl.argmax())
    cyc_fp32_total  = cyc0_fp32 + cyc1_fp32
    wall_fp32_total = wall0_fp32 + wall1_fp32

    mark = "\u2713" if pred_fp32 == label else "\u2717"
    print()
    print(f"  prediction: {pred_fp32} {mark}")
    print(f"  kernel 0: {cyc0_fp32:>7,} cycles ({cyc0_fp32 * 10:>10,} ns at 100 MHz)")
    print(f"  kernel 1: {cyc1_fp32:>7,} cycles ({cyc1_fp32 * 10:>10,} ns at 100 MHz)")
    print(f"  total:    {cyc_fp32_total:>7,} cycles ({cyc_fp32_total * 10:>10,} ns at 100 MHz)")
    print(f"  sim wall-clock: {wall_fp32_total:.1f}s")
    print()

    # ================================================================
    # Path C: INT8 quantized HDL simulation
    # ================================================================
    print("Path C \u2014 INT8 quantized HDL simulation:")
    print()

    with _noopt_scope(1):
        kernels_i8 = _compile_kernels(_build_int8_schedule())
    print(f"  compiled {len(kernels_i8)} kernels")
    print(f"  synthesising for ECP5 45F (Yosys + nextpnr-ecp5)...", end=" ", flush=True)
    stats_i8 = [_synthesis_stats(k) for k in kernels_i8]
    if stats_i8[0]["from_synth"]:
        print("done")
    else:
        print("tools not found, using RTLIL estimates")

    w1_q, w1_scale = quantize_int8(w1)
    x_q,  x_scale  = quantize_int8(x_float)
    w1_tg = w1_q.T.flatten()   # (784,128) int8 flattened
    b1_q  = np.round(b1 / (w1_scale * x_scale)).astype(np.int32)

    print(f"  running kernel 0 (128\u00d7784 GEMV + bias + ReLU)...", end=" ", flush=True)
    out1_i8, cyc0_i8, wall0_i8 = simulate_kernel(
        kernels_i8[0],
        {1: x_q, 2: w1_tg, 3: b1_q},
    )
    print(f"done ({wall0_i8:.1f}s)")

    hidden_i8 = out1_i8.astype(np.int8)

    w2_q, w2_scale = quantize_int8(w2)
    w2_tg = w2_q.T.flatten()   # (128,10) int8 flattened
    b2_q  = np.round(b2 / (w2_scale * 1.0)).astype(np.int32)

    print(f"  running kernel 1 (10\u00d7128 GEMV + bias)...", end=" ", flush=True)
    out2_i8, cyc1_i8, wall1_i8 = simulate_kernel(
        kernels_i8[1],
        {1: hidden_i8.astype(np.int32), 2: w2_tg, 3: b2_q},
    )
    print(f"done ({wall1_i8:.1f}s)")

    logits_i8_hdl = out2_i8.astype(np.float64) * w2_scale + b2
    pred_i8       = int(logits_i8_hdl.argmax())
    cyc_i8_total  = cyc0_i8 + cyc1_i8
    wall_i8_total = wall0_i8 + wall1_i8

    mark = "\u2713" if pred_i8 == label else "\u2717"
    print()
    print(f"  prediction: {pred_i8} {mark}")
    print(f"  kernel 0: {cyc0_i8:>7,} cycles ({cyc0_i8 * 10:>10,} ns at 100 MHz)")
    print(f"  kernel 1: {cyc1_i8:>7,} cycles ({cyc1_i8 * 10:>10,} ns at 100 MHz)")
    print(f"  total:    {cyc_i8_total:>7,} cycles ({cyc_i8_total * 10:>10,} ns at 100 MHz)")
    print(f"  sim wall-clock: {wall_i8_total:.1f}s")
    print()

    # ================================================================
    # Summary
    # ================================================================
    print("=" * 60)

    # --- Predictions ---
    preds = [pred_cpu, pred_fp32, pred_i8]
    if pred_gpu is not None:
        preds.append(pred_gpu)
    all_agree = len(set(preds)) == 1
    print(f"Predictions (true label: {label})")
    print(f"  CPU float32 : {pred_cpu}  {'\u2713' if pred_cpu == label else '\u2717'}")
    if pred_gpu is not None:
        print(f"  GPU {gpu_device:<8}: {pred_gpu}  {'\u2713' if pred_gpu == label else '\u2717'}")
    print(f"  FP32 HDL    : {pred_fp32}  {'\u2713' if pred_fp32 == label else '\u2717'}")
    print(f"  INT8 HDL    : {pred_i8}  {'\u2713' if pred_i8 == label else '\u2717'}")
    print(f"  All agree   : {'YES' if all_agree else 'NO'}")
    print()

    # --- Latency ---
    # Fmax: use minimum across all synthesised kernels (same clock domain)
    all_fmax = [s["fmax_mhz"] for s in stats_fp32 + stats_i8 if s["fmax_mhz"] is not None]
    fmax_mhz = min(all_fmax) if all_fmax else None

    cpu_ms     = t_cpu * 1e3
    hw_fp32_ms_100 = cyc_fp32_total / 100e6 * 1e3
    hw_i8_ms_100   = cyc_i8_total   / 100e6 * 1e3

    print(f"Inference latency (single image):")
    print(f"  CPU float32 : {cpu_ms:>8.3f} ms  (wall-clock, numpy/BLAS)")
    if gpu_ms is not None:
        print(f"  GPU {gpu_device:<8}: {gpu_ms:>8.3f} ms  (avg of 50 runs, float32)")
    else:
        print(f"  GPU         :      N/A     (no GPU detected)")

    if fmax_mhz is not None:
        hw_fp32_ms_fmax = cyc_fp32_total / (fmax_mhz * 1e6) * 1e3
        hw_i8_ms_fmax   = cyc_i8_total   / (fmax_mhz * 1e6) * 1e3
        print(f"  FP32 HDL    : {hw_fp32_ms_100:>8.3f} ms  at 100 MHz  "
              f"/ {hw_fp32_ms_fmax:.3f} ms  at {fmax_mhz:.0f} MHz  ({cyc_fp32_total:,} cycles)")
        print(f"  INT8 HDL    : {hw_i8_ms_100:>8.3f} ms  at 100 MHz  "
              f"/ {hw_i8_ms_fmax:.3f} ms  at {fmax_mhz:.0f} MHz  ({cyc_i8_total:,} cycles)")
    else:
        print(f"  FP32 HDL    : {hw_fp32_ms_100:>8.3f} ms  ({cyc_fp32_total:,} cycles at 100 MHz)")
        print(f"  INT8 HDL    : {hw_i8_ms_100:>8.3f} ms  ({cyc_i8_total:,} cycles at 100 MHz)")

    _ref_ms  = gpu_ms if gpu_ms is not None else cpu_ms
    _ref_lbl = f"GPU ({gpu_device})" if gpu_ms is not None else "CPU"
    _hdl_ms  = hw_fp32_ms_fmax if fmax_mhz is not None else hw_fp32_ms_100
    _ratio   = _ref_ms / _hdl_ms
    if _ratio < 1.0:
        print(f"  → {_ref_lbl} is {1/_ratio:.1f}× faster than FP32 HDL  "
              f"(sequential 1-MAC/cycle FSM — pipelining + batching needed to compete)")
    else:
        print(f"  → FP32 HDL is {_ratio:.1f}× faster than {_ref_lbl}")
    print()

    # --- Hardware resources (ECP5 45F synthesis) ---
    from_synth = stats_fp32[0]["from_synth"]
    src_label  = "ECP5 45F — Yosys + nextpnr-ecp5" if from_synth else "RTLIL pre-synthesis estimates"
    fp32_total_mem = sum(s["mem_bits"] for s in stats_fp32)
    i8_total_mem   = sum(s["mem_bits"] for s in stats_i8)
    fp32_ramb36    = math.ceil(fp32_total_mem / _RAMB36_BITS)
    i8_ramb36      = math.ceil(i8_total_mem   / _RAMB36_BITS)

    print(f"Hardware resources ({src_label}):")
    if from_synth:
        hdr = f"  {'':16s}  {'mem KB':>7s}  {'COMB':>6s}  {'FF':>5s}  {'DP16KD':>6s}  {'MULT18':>6s}  {'FP32 units':>10s}"
        print(hdr)
        for i, s in enumerate(stats_fp32):
            print(f"  float32 kernel {i}  {s['mem_bits']/8/1024:>7.1f}"
                  f"  {s['comb']:>6d}  {s['ff']:>5d}  {s['dp16kd']:>6d}  {s['mult18']:>6d}"
                  f"  {s['fp32_units']:>10d}")
        for i, s in enumerate(stats_i8):
            print(f"  int8    kernel {i}  {s['mem_bits']/8/1024:>7.1f}"
                  f"  {s['comb']:>6d}  {s['ff']:>5d}  {s['dp16kd']:>6d}  {s['mult18']:>6d}"
                  f"  {'N/A':>10s}")
        if fmax_mhz is not None:
            print(f"  worst-case Fmax: {fmax_mhz:.1f} MHz  (ECP5 45F, CABGA381, no pipeline regs)")
    else:
        print(f"  {'':16s}  {'mem':>8s}  {'FP32 units':>10s}")
        for i, s in enumerate(stats_fp32):
            print(f"  float32 kernel {i}  {s['mem_bits']/8/1024:>7.1f} KB  {s['fp32_units']:>10d}")
        for i, s in enumerate(stats_i8):
            print(f"  int8    kernel {i}  {s['mem_bits']/8/1024:>7.1f} KB  {'N/A':>10s}")
    print()
    print(f"  float32 total on-chip: {fp32_total_mem/8/1024:>6.1f} KB  → ~{fp32_ramb36} Xilinx RAMB36")
    print(f"  int8    total on-chip: {i8_total_mem/8/1024:>6.1f} KB  → ~{i8_ramb36} Xilinx RAMB36")
    print(f"  (dominated by weight matrices; streaming from SDRAM collapses")
    print(f"   on-chip BRAM to activation buffers only: ~1 KB per kernel)")


if __name__ == "__main__":
    main()
