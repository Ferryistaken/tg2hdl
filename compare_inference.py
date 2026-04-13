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
from compiler.utils import synthesis_stats
from tg2hdl.fpga_card import FPGACard, load_card


@contextmanager
def _noopt_scope(value=1):
    old = tg_noopt.value
    tg_noopt.value = value
    try:
        yield
    finally:
        tg_noopt.value = old


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

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="MNIST CPU/GPU/FPGA inference comparison")
    parser.add_argument("--img-idx", type=int, default=0, help="MNIST test index for single-image comparison")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for multi-image comparison")

    parser.add_argument("--cpu-warmup", type=int, default=3)
    parser.add_argument("--cpu-runs", type=int, default=20)
    parser.add_argument("--gpu-warmup", type=int, default=3)
    parser.add_argument("--gpu-runs", type=int, default=50)

    parser.add_argument("--skip-cpu", action="store_true", help="Disable CPU measurements")
    parser.add_argument("--skip-gpu", action="store_true", help="Disable GPU measurements")
    parser.add_argument("--skip-fp32", action="store_true", help="Disable FP32 FPGA path")
    parser.add_argument("--skip-int8", action="store_true", help="Disable INT8 FPGA path")
    parser.add_argument("--skip-stream", action="store_true", help="Disable multi-image FPGA stream simulation")
    parser.add_argument("--skip-synth", action="store_true", help="Skip synthesis estimation even if tools exist")
    return parser.parse_args()


def _load_data(img_idx: int, batch_size: int):
    state = safe_load("mnist_weights.safetensors")
    w1 = state["l1.weight"].numpy()
    b1 = state["l1.bias"].numpy()
    w2 = state["l2.weight"].numpy()
    b2 = state["l2.bias"].numpy()

    _, _, X_test, y_test = mnist()
    x_single = X_test[img_idx].numpy().reshape(1, 784).astype(np.float32) / 255.0
    label = int(y_test[img_idx].numpy())
    x_batch = X_test[:batch_size].numpy().reshape(batch_size, 784).astype(np.float32) / 255.0
    labels_batch = y_test[:batch_size].numpy().astype(np.int32)
    return (w1, b1, w2, b2), (x_single, label, x_batch, labels_batch)


def main():
    args = _parse_args()
    Tensor.training = False
    card = load_card()  # default FPGA card

    (w1, b1, w2, b2), (x_single, label, x_batch, labels_batch) = _load_data(args.img_idx, args.batch_size)
    x_float = x_single.reshape(784)

    print("=" * 60)
    print("MNIST Inference Comparison  (A=CPU · B=FP32-HDL · C=INT8-HDL)")
    print(f"FPGA card: {card.name}")
    print("=" * 60)
    print(f"Test image index: {args.img_idx}, true label: {label}")
    print(f"Config: batch={args.batch_size}, cpu_runs={args.cpu_runs}, gpu_runs={args.gpu_runs}")
    print()

    pred_cpu = pred_cpu_noopt = None
    t_cpu_ms = t_cpu_noopt_ms = None
    t_cpu_batch_ms = t_cpu_batch_noopt_ms = None

    pred_gpu_opt = pred_gpu_noopt = None
    gpu_ms_opt = gpu_ms_noopt = None
    gpu_batch_ms_opt = gpu_batch_ms_noopt = None
    gpu_device = None

    kernels_fp32 = []
    kernels_i8 = []
    stats_fp32 = []
    stats_i8 = []

    pred_fp32 = pred_i8 = None
    cyc_fp32_total = cyc_i8_total = None

    pred_fp32_stream = pred_i8_stream = None
    cyc_fp32_stream_total = cyc_i8_stream_total = None

    # CPU
    if not args.skip_cpu:
        pred_cpu_arr, t_cpu_ms = _cpu_inference(
            x_single, w1, b1, w2, b2, noopt=0, warmup=args.cpu_warmup, runs=args.cpu_runs
        )
        pred_cpu_noopt_arr, t_cpu_noopt_ms = _cpu_inference(
            x_single, w1, b1, w2, b2, noopt=1, warmup=args.cpu_warmup, runs=args.cpu_runs
        )
        pred_cpu = int(pred_cpu_arr[0])
        pred_cpu_noopt = int(pred_cpu_noopt_arr[0])

        pred_cpu_batch_arr, t_cpu_batch_ms = _cpu_inference(
            x_batch, w1, b1, w2, b2, noopt=0, warmup=args.cpu_warmup, runs=args.cpu_runs
        )
        pred_cpu_batch_noopt_arr, t_cpu_batch_noopt_ms = _cpu_inference(
            x_batch, w1, b1, w2, b2, noopt=1, warmup=args.cpu_warmup, runs=args.cpu_runs
        )
        batch_acc_opt = float((pred_cpu_batch_arr == labels_batch).mean())
        batch_acc_noopt = float((pred_cpu_batch_noopt_arr == labels_batch).mean())

        print("Path A — tinygrad float32 CPU:")
        print(f"  prediction (NOOPT=0): {pred_cpu} {'✓' if pred_cpu == label else '✗'}")
        print(f"  prediction (NOOPT=1): {pred_cpu_noopt} {'✓' if pred_cpu_noopt == label else '✗'}")
        print(f"  wall-clock (NOOPT=0): {t_cpu_ms:.3f} ms")
        print(f"  wall-clock (NOOPT=1): {t_cpu_noopt_ms:.3f} ms")
        print(f"  batch={args.batch_size} avg (NOOPT=0): {t_cpu_batch_ms:.3f} ms  ({t_cpu_batch_ms/args.batch_size:.3f} ms/img, acc={batch_acc_opt*100:.1f}%)")
        print(f"  batch={args.batch_size} avg (NOOPT=1): {t_cpu_batch_noopt_ms:.3f} ms  ({t_cpu_batch_noopt_ms/args.batch_size:.3f} ms/img, acc={batch_acc_noopt*100:.1f}%)")
        print()

    # GPU
    if not args.skip_gpu:
        print("Path G — tinygrad float32 GPU:", end=" ", flush=True)
        gpu_device = _detect_gpu()
        if gpu_device is None:
            print("no GPU detected (tried: " + ", ".join(_GPU_BACKENDS) + ")")
        else:
            print(f"{gpu_device} detected, running...", end=" ", flush=True)
            try:
                pred_gpu_opt_arr, gpu_ms_opt = _gpu_inference(
                    x_single, w1, b1, w2, b2, gpu_device, noopt=0, warmup=args.gpu_warmup, runs=args.gpu_runs
                )
                pred_gpu_noopt_arr, gpu_ms_noopt = _gpu_inference(
                    x_single, w1, b1, w2, b2, gpu_device, noopt=1, warmup=args.gpu_warmup, runs=args.gpu_runs
                )
                pred_gpu_opt = int(pred_gpu_opt_arr[0])
                pred_gpu_noopt = int(pred_gpu_noopt_arr[0])

                _, gpu_batch_ms_opt = _gpu_inference(
                    x_batch, w1, b1, w2, b2, gpu_device, noopt=0, warmup=args.gpu_warmup, runs=args.gpu_runs
                )
                _, gpu_batch_ms_noopt = _gpu_inference(
                    x_batch, w1, b1, w2, b2, gpu_device, noopt=1, warmup=args.gpu_warmup, runs=args.gpu_runs
                )
                print("done")
                print(f"  prediction (NOOPT=0): {pred_gpu_opt} {'✓' if pred_gpu_opt == label else '✗'}")
                print(f"  prediction (NOOPT=1): {pred_gpu_noopt} {'✓' if pred_gpu_noopt == label else '✗'}")
                print(f"  avg runs (NOOPT=0): {gpu_ms_opt:.3f} ms  (single image)")
                print(f"  avg runs (NOOPT=1): {gpu_ms_noopt:.3f} ms  (single image)")
                print(f"  batch={args.batch_size} avg (NOOPT=0): {gpu_batch_ms_opt:.3f} ms  ({gpu_batch_ms_opt/args.batch_size:.3f} ms/img)")
                print(f"  batch={args.batch_size} avg (NOOPT=1): {gpu_batch_ms_noopt:.3f} ms  ({gpu_batch_ms_noopt/args.batch_size:.3f} ms/img)")
            except Exception as e:
                print(f"failed ({e})")
                gpu_device = None
        print()

    if (not args.skip_fp32) or (not args.skip_int8):
        print("Compiling FPGA kernels...")

    # FP32 FPGA
    if not args.skip_fp32:
        print("Path B — float32 FPGA simulation:")
        with _noopt_scope(1):
            kernels_fp32 = _compile_kernels(_build_fp32_schedule())
        print(f"  compiled {len(kernels_fp32)} kernels")
        if args.skip_synth:
            stats_fp32 = [synthesis_stats(k, card=card)
                          | {"from_synth": False} for k in kernels_fp32]
            print("  synthesis stats skipped by option")
        else:
            print(f"  synthesising for {card.fpga_target_label()}...", end=" ", flush=True)
            stats_fp32 = [synthesis_stats(k, card=card)
                          for k in kernels_fp32]
            print("done" if stats_fp32[0]["from_synth"] else "tools not found, using RTLIL estimates")

        w1_fp32 = w1.T.astype(np.float32)
        w2_fp32 = w2.T.astype(np.float32)
        out1_fp32, cyc0_fp32, wall0_fp32 = simulate_kernel(kernels_fp32[0], {1: x_float, 2: w1_fp32.flatten(), 3: b1})
        hidden_fp32 = out1_fp32.astype(np.uint32).view(np.float32)
        out2_fp32, cyc1_fp32, wall1_fp32 = simulate_kernel(kernels_fp32[1], {1: hidden_fp32, 2: w2_fp32.flatten(), 3: b2})
        logits_fp32_hdl = out2_fp32.astype(np.uint32).view(np.float32)
        pred_fp32 = int(logits_fp32_hdl.argmax())
        cyc_fp32_total = cyc0_fp32 + cyc1_fp32
        print(f"  prediction: {pred_fp32} {'✓' if pred_fp32 == label else '✗'}")
        print(f"  total cycles: {cyc_fp32_total:,}  sim wall: {wall0_fp32 + wall1_fp32:.1f}s")
        print()

    # INT8 FPGA
    if not args.skip_int8:
        print("Path C — INT8 FPGA simulation:")
        with _noopt_scope(1):
            kernels_i8 = _compile_kernels(_build_int8_schedule())
        print(f"  compiled {len(kernels_i8)} kernels")
        if args.skip_synth:
            stats_i8 = [synthesis_stats(k, card=card)
                        | {"from_synth": False} for k in kernels_i8]
            print("  synthesis stats skipped by option")
        else:
            print(f"  synthesising for {card.fpga_target_label()}...", end=" ", flush=True)
            stats_i8 = [synthesis_stats(k, card=card)
                        for k in kernels_i8]
            print("done" if stats_i8[0]["from_synth"] else "tools not found, using RTLIL estimates")

        w1_q, w1_scale = quantize_int8(w1)
        x_q, x_scale = quantize_int8(x_float)
        w1_tg = w1_q.T.flatten()
        b1_q = np.round(b1 / (w1_scale * x_scale)).astype(np.int32)
        out1_i8, cyc0_i8, wall0_i8 = simulate_kernel(kernels_i8[0], {1: x_q, 2: w1_tg, 3: b1_q})
        hidden_i8 = out1_i8.astype(np.int8)

        w2_q, w2_scale = quantize_int8(w2)
        w2_tg = w2_q.T.flatten()
        b2_q = np.round(b2 / (w2_scale * 1.0)).astype(np.int32)
        out2_i8, cyc1_i8, wall1_i8 = simulate_kernel(kernels_i8[1], {1: hidden_i8.astype(np.int32), 2: w2_tg, 3: b2_q})
        logits_i8_hdl = out2_i8.astype(np.float64) * w2_scale + b2
        pred_i8 = int(logits_i8_hdl.argmax())
        cyc_i8_total = cyc0_i8 + cyc1_i8
        print(f"  prediction: {pred_i8} {'✓' if pred_i8 == label else '✗'}")
        print(f"  total cycles: {cyc_i8_total:,}  sim wall: {wall0_i8 + wall1_i8:.1f}s")
        print()

    if (not args.skip_stream) and kernels_fp32:
        print(f"Path S — FP32 FPGA stream simulation over N={args.batch_size} images...")
        pred_fp32_stream, cyc_fp32_stream_total, wall_fp32_stream_total = _simulate_fp32_stream(kernels_fp32, x_batch, w1, b1, w2, b2)
        print(f"  done in {wall_fp32_stream_total:.1f}s, acc={(pred_fp32_stream == labels_batch).mean()*100:.1f}%")
    if (not args.skip_stream) and kernels_i8:
        print(f"Path S — INT8 FPGA stream simulation over N={args.batch_size} images...")
        pred_i8_stream, cyc_i8_stream_total, wall_i8_stream_total = _simulate_int8_stream(kernels_i8, x_batch, w1, b1, w2, b2)
        print(f"  done in {wall_i8_stream_total:.1f}s, acc={(pred_i8_stream == labels_batch).mean()*100:.1f}%")
    if not args.skip_stream:
        print()

    print("=" * 60)
    print("Latency summary")
    if t_cpu_ms is not None:
        print(f"  CPU single NOOPT=0/1: {t_cpu_ms:.3f} / {t_cpu_noopt_ms:.3f} ms")
        print(f"  CPU batch  NOOPT=0/1: {t_cpu_batch_ms:.3f} / {t_cpu_batch_noopt_ms:.3f} ms")
    if gpu_ms_opt is not None:
        print(f"  GPU single NOOPT=0/1: {gpu_ms_opt:.3f} / {gpu_ms_noopt:.3f} ms")
        print(f"  GPU batch  NOOPT=0/1: {gpu_batch_ms_opt:.3f} / {gpu_batch_ms_noopt:.3f} ms")

    all_stats = stats_fp32 + stats_i8
    all_fmax = [s["fmax_mhz"] for s in all_stats if s["fmax_mhz"] is not None]
    fmax_mhz = min(all_fmax) if all_fmax else None

    if cyc_fp32_total is not None:
        ms100 = cyc_fp32_total / 100e6 * 1e3
        print(f"  FP32 FPGA single: {ms100:.3f} ms @100MHz ({cyc_fp32_total:,} cyc)")
    if cyc_i8_total is not None:
        ms100 = cyc_i8_total / 100e6 * 1e3
        print(f"  INT8 FPGA single: {ms100:.3f} ms @100MHz ({cyc_i8_total:,} cyc)")
    if cyc_fp32_stream_total is not None:
        ms100 = cyc_fp32_stream_total / 100e6 * 1e3
        print(f"  FP32 FPGA stream: {ms100:.3f} ms @100MHz ({cyc_fp32_stream_total:,} cyc)")
    if cyc_i8_stream_total is not None:
        ms100 = cyc_i8_stream_total / 100e6 * 1e3
        print(f"  INT8 FPGA stream: {ms100:.3f} ms @100MHz ({cyc_i8_stream_total:,} cyc)")

    if fmax_mhz is not None:
        print(f"  Synthesized worst-case Fmax: {fmax_mhz:.1f} MHz")

    if stats_fp32 or stats_i8:
        from_synth = (stats_fp32[0]["from_synth"] if stats_fp32 else stats_i8[0]["from_synth"])
        src_label = (f"{card.fpga_target_label()} — {card.synth_toolchain}"
                     if from_synth else "RTLIL pre-synthesis estimates")
        fp32_total_mem = sum(s["mem_bits"] for s in stats_fp32)
        i8_total_mem = sum(s["mem_bits"] for s in stats_i8)
        bram_type = card.bram_type
        bits_per_block = card.bram_data_bits_per_block
        fp32_blocks = math.ceil(fp32_total_mem / bits_per_block) if fp32_total_mem else 0
        i8_blocks = math.ceil(i8_total_mem / bits_per_block) if i8_total_mem else 0
        print()
        print(f"Hardware resources ({src_label}):")
        print(f"  float32 total on-chip: {fp32_total_mem/8/1024:>6.1f} KB  → ~{fp32_blocks} {bram_type}")
        print(f"  int8    total on-chip: {i8_total_mem/8/1024:>6.1f} KB  → ~{i8_blocks} {bram_type}")


if __name__ == "__main__":
    main()
