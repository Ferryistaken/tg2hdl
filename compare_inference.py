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


def _cpu_inference(x_input, w1, b1, w2, b2, *, noopt: int, warmup: int = 3, runs: int = 20):
    """Run CPU forward pass(es) under a specific NOOPT setting.

    Returns (predictions, avg_ms) where predictions is a 1D numpy array.
    """
    with _noopt_scope(noopt):
        x_t = Tensor(x_input)
        w1_t, b1_t = Tensor(w1), Tensor(b1)
        w2_t, b2_t = Tensor(w2), Tensor(b2)

        def _forward():
            h = (x_t.linear(w1_t.T, b1_t)).relu()
            logits = h.linear(w2_t.T, b2_t)
            return logits.numpy()

        for _ in range(warmup):
            _forward()

        t0 = time.perf_counter()
        for _ in range(runs):
            out = _forward()
        avg_ms = (time.perf_counter() - t0) / runs * 1e3

    preds = np.asarray(out.argmax(axis=1), dtype=np.int32)
    return preds, avg_ms


def _gpu_inference(x_input, w1, b1, w2, b2, device, *, noopt: int, warmup=3, runs=50):
    """Run GPU forward pass(es) under a specific NOOPT setting.

    Returns (predictions, avg_ms) where predictions is a 1D numpy array.
    """
    with _noopt_scope(noopt):
        x_t = Tensor(x_input, device=device)
        w1_t = Tensor(w1, device=device)
        b1_t = Tensor(b1, device=device)
        w2_t = Tensor(w2, device=device)
        b2_t = Tensor(b2, device=device)

        def _forward():
            h = (x_t.linear(w1_t.T, b1_t)).relu()
            logits = h.linear(w2_t.T, b2_t)
            return logits.numpy()

        for _ in range(warmup):
            _forward()

        t0 = time.perf_counter()
        for _ in range(runs):
            out = _forward()
        avg_ms = (time.perf_counter() - t0) / runs * 1e3

    preds = np.asarray(out.argmax(axis=1), dtype=np.int32)
    return preds, avg_ms


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


def _simulate_fp32_stream(kernels_fp32, x_batch, w1, b1, w2, b2):
    """Simulate batch_size images sequentially through FP32 FPGA kernels."""
    w1_fp32 = w1.T.astype(np.float32).flatten()
    w2_fp32 = w2.T.astype(np.float32).flatten()

    preds = []
    total_cycles = 0
    total_wall = 0.0
    for i in range(x_batch.shape[0]):
        out1_fp32, cyc0, wall0 = simulate_kernel(
            kernels_fp32[0],
            {1: x_batch[i], 2: w1_fp32, 3: b1},
        )
        hidden_fp32 = out1_fp32.astype(np.uint32).view(np.float32)

        out2_fp32, cyc1, wall1 = simulate_kernel(
            kernels_fp32[1],
            {1: hidden_fp32, 2: w2_fp32, 3: b2},
        )
        logits = out2_fp32.astype(np.uint32).view(np.float32)
        preds.append(int(logits.argmax()))
        total_cycles += cyc0 + cyc1
        total_wall += wall0 + wall1

    return np.asarray(preds, dtype=np.int32), total_cycles, total_wall


def _simulate_int8_stream(kernels_i8, x_batch, w1, b1, w2, b2):
    """Simulate batch_size images sequentially through INT8 FPGA kernels."""
    w1_q, w1_scale = quantize_int8(w1)
    w2_q, w2_scale = quantize_int8(w2)
    w1_tg = w1_q.T.flatten()
    w2_tg = w2_q.T.flatten()

    preds = []
    total_cycles = 0
    total_wall = 0.0
    for i in range(x_batch.shape[0]):
        x_q, x_scale = quantize_int8(x_batch[i])
        b1_q = np.round(b1 / (w1_scale * x_scale)).astype(np.int32)

        out1_i8, cyc0, wall0 = simulate_kernel(
            kernels_i8[0],
            {1: x_q, 2: w1_tg, 3: b1_q},
        )
        hidden_i8 = out1_i8.astype(np.int8)

        b2_q = np.round(b2 / (w2_scale * 1.0)).astype(np.int32)
        out2_i8, cyc1, wall1 = simulate_kernel(
            kernels_i8[1],
            {1: hidden_i8.astype(np.int32), 2: w2_tg, 3: b2_q},
        )

        logits = out2_i8.astype(np.float64) * w2_scale + b2
        preds.append(int(logits.argmax()))
        total_cycles += cyc0 + cyc1
        total_wall += wall0 + wall1

    return np.asarray(preds, dtype=np.int32), total_cycles, total_wall


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

    # --- Test image + batch ---
    _, _, X_test, y_test = mnist()
    img_idx = 0
    batch_size = 64
    x_single = X_test[img_idx].numpy().reshape(1, 784).astype(np.float32) / 255.0
    label = int(y_test[img_idx].numpy())
    x_batch = X_test[:batch_size].numpy().reshape(batch_size, 784).astype(np.float32) / 255.0
    labels_batch = y_test[:batch_size].numpy().astype(np.int32)
    x_float = x_single.reshape(784)

    print("=" * 60)
    print("MNIST Inference Comparison  (A=CPU · B=FP32-HDL · C=INT8-HDL)")
    print("=" * 60)
    print(f"Test image index: {img_idx}, true label: {label}")
    print()

    # ================================================================
    # Path A: tinygrad float32 CPU (optimized + NOOPT)
    # ================================================================
    Tensor.training = False
    pred_cpu_arr, t_cpu_ms = _cpu_inference(x_single, w1, b1, w2, b2, noopt=0)
    pred_cpu_noopt_arr, t_cpu_noopt_ms = _cpu_inference(x_single, w1, b1, w2, b2, noopt=1)
    pred_cpu = int(pred_cpu_arr[0])
    pred_cpu_noopt = int(pred_cpu_noopt_arr[0])

    pred_cpu_batch_arr, t_cpu_batch_ms = _cpu_inference(x_batch, w1, b1, w2, b2, noopt=0)
    pred_cpu_batch_noopt_arr, t_cpu_batch_noopt_ms = _cpu_inference(x_batch, w1, b1, w2, b2, noopt=1)
    batch_acc_opt = float((pred_cpu_batch_arr == labels_batch).mean())
    batch_acc_noopt = float((pred_cpu_batch_noopt_arr == labels_batch).mean())

    mark = "✓" if pred_cpu == label else "✗"
    mark_noopt = "✓" if pred_cpu_noopt == label else "✗"
    print(f"Path A — tinygrad float32 CPU:")
    print(f"  prediction (NOOPT=0): {pred_cpu} {mark}")
    print(f"  prediction (NOOPT=1): {pred_cpu_noopt} {mark_noopt}")
    print(f"  wall-clock (NOOPT=0): {t_cpu_ms:.3f} ms")
    print(f"  wall-clock (NOOPT=1): {t_cpu_noopt_ms:.3f} ms")
    print(f"  batch={batch_size} avg (NOOPT=0): {t_cpu_batch_ms:.3f} ms  ({t_cpu_batch_ms/batch_size:.3f} ms/img, acc={batch_acc_opt*100:.1f}%)")
    print(f"  batch={batch_size} avg (NOOPT=1): {t_cpu_batch_noopt_ms:.3f} ms  ({t_cpu_batch_noopt_ms/batch_size:.3f} ms/img, acc={batch_acc_noopt*100:.1f}%)")
    print()

    # ================================================================
    # Path G: GPU (if available)
    # ================================================================
    print("Path G — tinygrad float32 GPU:", end=" ", flush=True)
    gpu_device = _detect_gpu()
    pred_gpu_opt = gpu_ms_opt = None
    pred_gpu_noopt = gpu_ms_noopt = None
    gpu_batch_ms_opt = gpu_batch_ms_noopt = None
    pred_gpu_batch_opt = pred_gpu_batch_noopt = None
    if gpu_device is None:
        print("no GPU detected (tried: " + ", ".join(_GPU_BACKENDS) + ")")
    else:
        print(f"{gpu_device} detected, running...", end=" ", flush=True)
        try:
            pred_gpu_opt_arr, gpu_ms_opt = _gpu_inference(x_single, w1, b1, w2, b2, gpu_device, noopt=0)
            pred_gpu_noopt_arr, gpu_ms_noopt = _gpu_inference(x_single, w1, b1, w2, b2, gpu_device, noopt=1)
            pred_gpu_opt = int(pred_gpu_opt_arr[0])
            pred_gpu_noopt = int(pred_gpu_noopt_arr[0])

            pred_gpu_batch_opt, gpu_batch_ms_opt = _gpu_inference(x_batch, w1, b1, w2, b2, gpu_device, noopt=0)
            pred_gpu_batch_noopt, gpu_batch_ms_noopt = _gpu_inference(x_batch, w1, b1, w2, b2, gpu_device, noopt=1)

            mark_opt = "✓" if pred_gpu_opt == label else "✗"
            mark_noopt = "✓" if pred_gpu_noopt == label else "✗"
            batch_acc_gpu_opt = float((pred_gpu_batch_opt == labels_batch).mean())
            batch_acc_gpu_noopt = float((pred_gpu_batch_noopt == labels_batch).mean())
            print("done")
            print(f"  prediction (NOOPT=0): {pred_gpu_opt} {mark_opt}")
            print(f"  prediction (NOOPT=1): {pred_gpu_noopt} {mark_noopt}")
            print(f"  avg of 50 runs (NOOPT=0): {gpu_ms_opt:.3f} ms  (single image)")
            print(f"  avg of 50 runs (NOOPT=1): {gpu_ms_noopt:.3f} ms  (single image)")
            print(f"  batch={batch_size} avg (NOOPT=0): {gpu_batch_ms_opt:.3f} ms  ({gpu_batch_ms_opt/batch_size:.3f} ms/img, acc={batch_acc_gpu_opt*100:.1f}%)")
            print(f"  batch={batch_size} avg (NOOPT=1): {gpu_batch_ms_noopt:.3f} ms  ({gpu_batch_ms_noopt/batch_size:.3f} ms/img, acc={batch_acc_gpu_noopt*100:.1f}%)")
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
    # Multi-image FPGA streaming simulation (actual, not scaled)
    # ================================================================
    print(f"Path S — FPGA streaming simulation over N={batch_size} images:")
    print("  Note: this runs full sequential simulation for each image and can take a long time.")

    print("  running FP32 stream...", end=" ", flush=True)
    pred_fp32_stream, cyc_fp32_stream_total, wall_fp32_stream_total = _simulate_fp32_stream(
        kernels_fp32, x_batch, w1, b1, w2, b2
    )
    fp32_stream_acc = float((pred_fp32_stream == labels_batch).mean())
    print(f"done ({wall_fp32_stream_total:.1f}s)")

    print("  running INT8 stream...", end=" ", flush=True)
    pred_i8_stream, cyc_i8_stream_total, wall_i8_stream_total = _simulate_int8_stream(
        kernels_i8, x_batch, w1, b1, w2, b2
    )
    i8_stream_acc = float((pred_i8_stream == labels_batch).mean())
    print(f"done ({wall_i8_stream_total:.1f}s)")
    print()

    # ================================================================
    # Summary
    # ================================================================
    print("=" * 60)

    # --- Predictions ---
    preds = [pred_cpu, pred_cpu_noopt, pred_fp32, pred_i8]
    if pred_gpu_opt is not None:
        preds.extend([pred_gpu_opt, pred_gpu_noopt])
    all_agree = len(set(preds)) == 1
    print(f"Predictions (true label: {label})")
    print(f"  CPU float32 NOOPT=0: {pred_cpu}  {'\u2713' if pred_cpu == label else '\u2717'}")
    print(f"  CPU float32 NOOPT=1: {pred_cpu_noopt}  {'\u2713' if pred_cpu_noopt == label else '\u2717'}")
    if pred_gpu_opt is not None:
        print(f"  GPU {gpu_device:<8} NOOPT=0: {pred_gpu_opt}  {'\u2713' if pred_gpu_opt == label else '\u2717'}")
        print(f"  GPU {gpu_device:<8} NOOPT=1: {pred_gpu_noopt}  {'\u2713' if pred_gpu_noopt == label else '\u2717'}")
    print(f"  FP32 HDL    : {pred_fp32}  {'\u2713' if pred_fp32 == label else '\u2717'}")
    print(f"  INT8 HDL    : {pred_i8}  {'\u2713' if pred_i8 == label else '\u2717'}")
    print(f"  All agree   : {'YES' if all_agree else 'NO'}")
    print()

    # --- Latency ---
    # Fmax: use minimum across all synthesised kernels (same clock domain)
    all_fmax = [s["fmax_mhz"] for s in stats_fp32 + stats_i8 if s["fmax_mhz"] is not None]
    fmax_mhz = min(all_fmax) if all_fmax else None

    cpu_ms = t_cpu_ms
    cpu_noopt_ms = t_cpu_noopt_ms
    cpu_batch_ms = t_cpu_batch_ms
    cpu_batch_noopt_ms = t_cpu_batch_noopt_ms

    hw_fp32_ms_100 = cyc_fp32_total / 100e6 * 1e3
    hw_i8_ms_100   = cyc_i8_total   / 100e6 * 1e3
    hw_fp32_stream_ms_100 = cyc_fp32_stream_total / 100e6 * 1e3
    hw_i8_stream_ms_100   = cyc_i8_stream_total / 100e6 * 1e3

    if fmax_mhz is not None:
        hw_fp32_ms_fmax = cyc_fp32_total / (fmax_mhz * 1e6) * 1e3
        hw_i8_ms_fmax   = cyc_i8_total   / (fmax_mhz * 1e6) * 1e3
        hw_fp32_stream_ms_fmax = cyc_fp32_stream_total / (fmax_mhz * 1e6) * 1e3
        hw_i8_stream_ms_fmax   = cyc_i8_stream_total / (fmax_mhz * 1e6) * 1e3
    else:
        hw_fp32_ms_fmax = hw_i8_ms_fmax = None
        hw_fp32_stream_ms_fmax = hw_i8_stream_ms_fmax = None

    print("Single-image latency comparison:")
    print(f"  CPU float32 NOOPT=0: {cpu_ms:>8.3f} ms")
    print(f"  CPU float32 NOOPT=1: {cpu_noopt_ms:>8.3f} ms")
    if gpu_ms_opt is not None:
        print(f"  GPU {gpu_device:<8} NOOPT=0: {gpu_ms_opt:>8.3f} ms")
        print(f"  GPU {gpu_device:<8} NOOPT=1: {gpu_ms_noopt:>8.3f} ms")
    else:
        print(f"  GPU         :      N/A     (no GPU detected)")

    if fmax_mhz is not None:
        print(f"  FP32 FPGA   : {hw_fp32_ms_100:>8.3f} ms at 100 MHz / {hw_fp32_ms_fmax:.3f} ms at {fmax_mhz:.0f} MHz")
        print(f"  INT8 FPGA   : {hw_i8_ms_100:>8.3f} ms at 100 MHz / {hw_i8_ms_fmax:.3f} ms at {fmax_mhz:.0f} MHz")
    else:
        print(f"  FP32 FPGA   : {hw_fp32_ms_100:>8.3f} ms at 100 MHz")
        print(f"  INT8 FPGA   : {hw_i8_ms_100:>8.3f} ms at 100 MHz")
    print()

    print(f"Multi-image comparison (N={batch_size}):")
    print(f"  CPU batch NOOPT=0: {cpu_batch_ms:>8.3f} ms  ({cpu_batch_ms/batch_size:.3f} ms/img)")
    print(f"  CPU batch NOOPT=1: {cpu_batch_noopt_ms:>8.3f} ms  ({cpu_batch_noopt_ms/batch_size:.3f} ms/img)")
    if gpu_ms_opt is not None:
        print(f"  GPU batch NOOPT=0: {gpu_batch_ms_opt:>8.3f} ms  ({gpu_batch_ms_opt/batch_size:.3f} ms/img)")
        print(f"  GPU batch NOOPT=1: {gpu_batch_ms_noopt:>8.3f} ms  ({gpu_batch_ms_noopt/batch_size:.3f} ms/img)")
    else:
        print(f"  GPU batch      :      N/A     (no GPU detected)")

    if fmax_mhz is not None:
        print(f"  FP32 FPGA stream: {hw_fp32_stream_ms_100:>8.3f} ms at 100 MHz / {hw_fp32_stream_ms_fmax:.3f} ms at {fmax_mhz:.0f} MHz")
        print(f"  INT8 FPGA stream: {hw_i8_stream_ms_100:>8.3f} ms at 100 MHz / {hw_i8_stream_ms_fmax:.3f} ms at {fmax_mhz:.0f} MHz")
    else:
        print(f"  FP32 FPGA stream: {hw_fp32_stream_ms_100:>8.3f} ms at 100 MHz")
        print(f"  INT8 FPGA stream: {hw_i8_stream_ms_100:>8.3f} ms at 100 MHz")

    single_ref_ms = gpu_ms_opt if gpu_ms_opt is not None else cpu_ms
    single_ref_lbl = f"GPU ({gpu_device}, NOOPT=0)" if gpu_ms_opt is not None else "CPU (NOOPT=0)"
    single_fpga_ms = hw_fp32_ms_fmax if hw_fp32_ms_fmax is not None else hw_fp32_ms_100
    single_ratio = single_ref_ms / single_fpga_ms

    multi_ref_ms = gpu_batch_ms_opt if gpu_batch_ms_opt is not None else cpu_batch_ms
    multi_ref_lbl = f"GPU batch ({gpu_device}, NOOPT=0)" if gpu_batch_ms_opt is not None else "CPU batch (NOOPT=0)"
    multi_fpga_ms = hw_fp32_stream_ms_fmax if hw_fp32_stream_ms_fmax is not None else hw_fp32_stream_ms_100
    multi_ratio = multi_ref_ms / multi_fpga_ms

    if single_ratio < 1.0:
        print(f"  → Single-image: {single_ref_lbl} is {1/single_ratio:.1f}× faster than FP32 FPGA")
    else:
        print(f"  → Single-image: FP32 FPGA is {single_ratio:.1f}× faster than {single_ref_lbl}")

    if multi_ratio < 1.0:
        print(f"  → Multi-image: {multi_ref_lbl} is {1/multi_ratio:.1f}× faster than FP32 FPGA stream")
    else:
        print(f"  → Multi-image: FP32 FPGA stream is {multi_ratio:.1f}× faster than {multi_ref_lbl}")

    print("  [note] FPGA stream uses actual repeated simulation per image; wall-clock includes simulator overhead.")
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
