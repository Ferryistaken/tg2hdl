"""Demo: when does CPU beat the FPGA?

Short answer: always on elementwise/reduction ops that fit in CPU cache,
because a modern CPU does 8-32 ops/cycle (AVX/SIMD) at 3+ GHz, while the
current HDL backend does exactly 1 op/cycle at ~100 MHz.

  CPU (AVX2, int32): 8 ops/cycle × 3 GHz = 24 Gop/s
  FPGA (current):    1 op/cycle × 100 MHz =  0.1 Gop/s  →  240× slower

The FPGA only *looks* faster when measuring via tg2hdl.benchmark() against
tinygrad, because tinygrad's Python overhead inflates the CPU baseline.
This demo exposes that by also timing tight numpy loops.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from tinygrad import Tensor, dtypes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("NOOPT", "1")
os.environ.setdefault("DEBUG", "0")

import tg2hdl
from compiler import compile_top_module
from compiler.top_module import simulate_top
from compiler.visualize import analyze_schedule
from tg2hdl.report import _infer_input_data

FMAX_MHZ = 100.0


def numpy_time_us(fn, warmup=10, iters=200) -> float:
    """Tight-loop numpy timing in µs — bypasses Python/tinygrad overhead."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e6


def fpga_time_us(cycles: int) -> float:
    return cycles / (FMAX_MHZ * 1e6) * 1e6


def run_hdl(expr, schedule=None):
    """Simulate via TopModule, return cycle_counts."""
    if schedule is None:
        schedule = expr.schedule()
    pv = analyze_schedule(schedule)
    top, _, _ = compile_top_module(schedule)
    input_data = _infer_input_data(schedule, pv, top)
    _, cc, _ = simulate_top(top, input_data)
    return cc


def section(title):
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def row(label, cpu_us, fpga_us):
    ratio = fpga_us / cpu_us if cpu_us > 0 else float("inf")
    winner = "CPU" if ratio > 1 else "FPGA"
    print(f"  {label:<30}  cpu: {cpu_us:>8.2f} µs  fpga: {fpga_us:>8.2f} µs  "
          f"ratio: {ratio:>6.1f}×  winner: {winner}")


rng = np.random.RandomState(42)

# ──────────────────────────────────────────────────────────────
# 1. Elementwise add — pure SIMD on CPU, 1-op/cycle on FPGA
# ──────────────────────────────────────────────────────────────
section("Elementwise int32 add  (vectorizable — CPU wins)")

for N in [64, 512, 4096]:
    x_np = rng.randint(-100, 100, (N,)).astype(np.int32)
    c_np = rng.randint(-100, 100, (N,)).astype(np.int32)

    cpu_us = numpy_time_us(lambda: x_np + c_np)

    x_t = Tensor(x_np); c_t = Tensor(c_np)
    cc = run_hdl(x_t + c_t)
    fpga_us_ = fpga_time_us(cc["total"])

    row(f"add({N},)", cpu_us, fpga_us_)

# ──────────────────────────────────────────────────────────────
# 2. INT8 matmul — still 1 MAC/cycle, but CPU is also slow here
# ──────────────────────────────────────────────────────────────
section("INT8 matmul  (1×K @ K×1 dot product — FPGA may win?)")

for K in [4, 32, 256]:
    x_np = rng.randint(-4, 4, (1, K)).astype(np.int8)
    w_np = rng.randint(-3, 3, (K, 1)).astype(np.int8)

    cpu_us = numpy_time_us(
        lambda: (x_np.astype(np.int32) @ w_np.astype(np.int32)).astype(np.int32)
    )

    x_t = Tensor(x_np); w_t = Tensor(w_np)
    cc = run_hdl((x_t @ w_t).cast(dtypes.int32))
    fpga_us_ = fpga_time_us(cc["total"])

    row(f"matmul(1,{K},{1})", cpu_us, fpga_us_)

# ──────────────────────────────────────────────────────────────
# 3. Full MLP layer (matmul + bias + relu) via tg2hdl.benchmark
# ──────────────────────────────────────────────────────────────
section("tg2hdl.benchmark() — tinygrad CPU vs projected FPGA+PCIe (Gen3 x4)")

x = Tensor(rng.randint(-4, 4, (1, 4)).astype(np.int8))
w = Tensor(rng.randint(-3, 3, (4, 3)).astype(np.int8))
b = Tensor(rng.randint(-10, 10, (1, 3)).astype(np.int32))
art = tg2hdl.benchmark((x @ w).cast(dtypes.int32) + b, out_dir="tmp/demo_report_vs")

cpu_us   = art.tinygrad_wall_s * 1e6
fpga_compute_us = fpga_time_us(art.tg2hdl_total_cycles)
pcie_in_us  = art.pcie_in_s  * 1e6 if art.pcie_in_s  else 0
pcie_out_us = art.pcie_out_s * 1e6 if art.pcie_out_s else 0
fpga_total_us = (art.fpga_with_pcie_s * 1e6) if art.fpga_with_pcie_s else (fpga_compute_us + pcie_in_us + pcie_out_us)

print(f"  CPU (tinygrad, incl. Python overhead):  {cpu_us:.1f} µs")
print(f"  FPGA compute only (cycles/fmax):        {fpga_compute_us:.2f} µs  ({art.tg2hdl_total_cycles} cycles)")
print(f"  PCIe in  ({art.pcie_in_s and art.pcie_in_s*1e6 or 0:.1f} µs) + out ({art.pcie_out_s and art.pcie_out_s*1e6 or 0:.1f} µs)")
print(f"  FPGA+PCIe total:                        {fpga_total_us:.2f} µs")
if fpga_total_us > 0:
    ratio = cpu_us / fpga_total_us
    print(f"  Ratio: {ratio:.1f}×  ({'FPGA+PCIe faster' if ratio > 1 else 'CPU faster'})")

# ──────────────────────────────────────────────────────────────
print()
print("Summary")
print("-------")
print("  Elementwise ops: CPU wins by 10-1000× (SIMD vs 1 op/cycle at 100 MHz)")
print("  Matmuls:         FPGA can win when problem is tiny (fmax × latency < overhead)")
print("  tinygrad bench:  misleading for CPU — includes Python/scheduling overhead")
print("  Fix for FPGA:    reduce_unroll_factor=N → N MACs/cycle, closes the gap")
