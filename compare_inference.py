"""Compare MNIST inference: tinygrad float32 CPU vs compiler-generated HDL simulation.

Path A: tinygrad float32 on CPU (reference).
Path B: tinygrad UOps → HDL compiler → Amaranth simulation (INT8 quantized).
"""

import os

os.environ["NOOPT"] = "1"
os.environ["DEBUG"] = "0"

import time
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import safe_load

from tinygrad.uop.ops import Ops

from compiler import HDLRenderer, compile_kernel, simulate_kernel, quantize_int8
from compiler.backend import _get_uops


def main():
    # --- Load trained weights ---
    state = safe_load("mnist_weights.safetensors")
    w1 = state["l1.weight"].numpy()   # (128, 784)
    b1 = state["l1.bias"].numpy()     # (128,)
    w2 = state["l2.weight"].numpy()   # (10, 128)
    b2 = state["l2.bias"].numpy()     # (10,)

    # --- Load a test image ---
    _, _, X_test, y_test = mnist()
    img_idx = 0
    x_float = X_test[img_idx].numpy().reshape(784).astype(np.float32) / 255.0
    label = int(y_test[img_idx].numpy())

    print("=" * 60)
    print("MNIST Inference Comparison")
    print("=" * 60)
    print(f"Test image index: {img_idx}, true label: {label}")
    print()

    # ============================================================
    # Path A: tinygrad float32 CPU
    # ============================================================
    x_t = Tensor(x_float.reshape(1, 784))
    w1_t, b1_t = Tensor(w1), Tensor(b1)
    w2_t, b2_t = Tensor(w2), Tensor(b2)

    Tensor.training = False
    t0 = time.perf_counter()
    h = (x_t.linear(w1_t.T, b1_t)).relu()
    logits = h.linear(w2_t.T, b2_t)
    logits_np = logits.numpy().flatten()
    pred_tinygrad = int(logits_np.argmax())
    t_tinygrad = time.perf_counter() - t0

    mark = "\u2713" if pred_tinygrad == label else "\u2717"
    print(f"tinygrad (float32 CPU):")
    print(f"  prediction: {pred_tinygrad} {mark}")
    print(f"  wall-clock: {t_tinygrad*1000:.2f} ms")
    print()

    # ============================================================
    # Path B: compiler-generated HDL simulation (INT8 quantized)
    # ============================================================
    print(f"Compiler-generated HDL simulation (INT8):")
    print()

    # --- Compile the model graph ---
    # Build the tinygrad graph matching our model architecture
    x_sym = Tensor.empty(1, 784, dtype=dtypes.int8)
    w1_sym = Tensor.empty(784, 128, dtype=dtypes.int8)
    b1_sym = Tensor.empty(1, 128, dtype=dtypes.int32)
    w2_sym = Tensor.empty(128, 10, dtype=dtypes.int8)
    b2_sym = Tensor.empty(1, 10, dtype=dtypes.int32)

    h_sym = ((x_sym @ w1_sym).cast(dtypes.int32) + b1_sym).relu()
    h_i8 = h_sym.cast(dtypes.int8)
    logits_sym = (h_i8 @ w2_sym).cast(dtypes.int32) + b2_sym
    sched = logits_sym.schedule()

    renderer = HDLRenderer()
    kernels = []
    for si in sched:
        if si.ast.op != Ops.SINK:
            continue
        uops = _get_uops(si.ast, renderer)
        kernels.append(compile_kernel(uops))

    print(f"  compiled {len(kernels)} kernels from tinygrad UOps")

    # --- Quantize weights ---
    w1_q, w1_scale = quantize_int8(w1)       # (128, 784) int8
    x_q, x_scale = quantize_int8(x_float)    # (784,) int8

    # Transpose w1 for tinygrad's matmul layout: x(1,784) @ w(784,128)
    # tinygrad stores as w[j*128 + i], so transpose from (128,784) to (784,128)
    w1_tg = w1_q.T.flatten()  # (784, 128) flattened

    # Quantize bias to int32 (scale = w_scale * x_scale)
    b1_q = np.round(b1 / (w1_scale * x_scale)).astype(np.int32)

    # --- Layer 1: GEMV(128, 784) + bias + ReLU ---
    print(f"  running kernel 0 (128x784 GEMV + bias + ReLU)...", end=" ", flush=True)
    t0 = time.perf_counter()
    out1, cycles1, wall1 = simulate_kernel(
        kernels[0],
        {1: x_q, 2: w1_tg, 3: b1_q},
    )
    print(f"done ({wall1:.1f}s)")

    # --- Layer 2: GEMV(10, 128) + bias ---
    # out1 is already int8-truncated by the kernel (cast + relu)
    # We need to re-quantize for layer 2
    # The kernel output is int32 but contains int8-range values from the casts
    hidden_int8 = out1.astype(np.int8)

    w2_q, w2_scale = quantize_int8(w2)
    w2_tg = w2_q.T.flatten()  # (128, 10) flattened

    # For layer 2 bias: scale is relative to hidden * w2 product
    # hidden is in int8 units (already quantized), w2 is quantized
    # The scale of hidden values is approximately w1_scale * x_scale (from the accumulator)
    # but they were truncated to int8 by the kernel, so we treat them as int8 with scale ~1
    # For simplicity, quantize bias relative to w2_scale
    h_scale = 1.0  # hidden values are already int8 from kernel output
    b2_q = np.round(b2 / (w2_scale * h_scale)).astype(np.int32)

    print(f"  running kernel 1 (10x128 GEMV + bias)...", end=" ", flush=True)
    out2, cycles2, wall2 = simulate_kernel(
        kernels[1],
        {1: hidden_int8.astype(np.int32), 2: w2_tg, 3: b2_q},
    )
    print(f"done ({wall2:.1f}s)")

    # --- Dequantize and predict ---
    # Output is int32 values; dequantize to float for argmax
    logits_hdl = out2.astype(np.float64) * w2_scale * h_scale + b2
    pred_hdl = int(logits_hdl.argmax())

    total_cycles = cycles1 + cycles2
    total_wall = wall1 + wall2

    mark = "\u2713" if pred_hdl == label else "\u2717"
    print()
    print(f"  prediction: {pred_hdl} {mark}")
    print(f"  kernel 0: {cycles1:>7,} cycles ({cycles1 * 10:>10,} ns at 100 MHz)")
    print(f"  kernel 1: {cycles2:>7,} cycles ({cycles2 * 10:>10,} ns at 100 MHz)")
    print(f"  total:    {total_cycles:>7,} cycles ({total_cycles * 10:>10,} ns at 100 MHz)")
    print(f"  sim wall-clock: {total_wall:.1f}s (Python simulation overhead)")

    # --- Summary ---
    print()
    print("-" * 60)
    agree = "AGREE" if pred_tinygrad == pred_hdl else "DISAGREE"
    print(f"Predictions {agree}: tinygrad={pred_tinygrad}, HDL={pred_hdl}, true={label}")
    print(f"Simulated HW time: {total_cycles * 10 / 1e6:.3f} ms at 100 MHz")


if __name__ == "__main__":
    main()
