"""Performance benchmark suite — 10 workloads across complexity tiers.

Each test:
  1. Asserts correctness (HDL output == tinygrad CPU reference).
  2. Prints cycle count and timing information.
  3. Validates the cycle count against the analytical model where applicable.

Tiers
-----
  Simple       — scalar, elementwise (tests 1–3)
  Composed     — single-kernel GEMV and compound ops (tests 4–6)
  Multi-kernel — chained kernel execution (tests 7–8)
  Float        — float32 dtype, analytical-only path (tests 9)
  ML-scale     — MNIST-like shapes (test 10, marked slow)

Run all (except slow):
    uv run pytest benchmarks/perf_suite.py -v -s -k "not slow"

Run everything:
    uv run pytest benchmarks/perf_suite.py -v -s
"""

import os

os.environ.setdefault("DEBUG", "0")

import numpy as np
import pytest
from tinygrad import Tensor, dtypes

from benchmarks.harness import run_bench

rng = np.random.RandomState(2025)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_gemv_cycles(M: int, K: int) -> int:
    """M*(K+2)+1 — FSM cycle model for a 2-level GEMV kernel."""
    return M * (K + 2) + 1


def _check(r, *, cycles_exact: int | None = None, cycles_approx: int | None = None,
           tol: int = 2):
    """Assert correctness and optionally verify cycle count."""
    assert r.correct, (
        f"[{r.name}] FAIL  max_err={r.max_abs_error}\n"
        f"  ref={r.output_ref}\n  hdl={r.output_hdl}"
    )
    if cycles_exact is not None:
        assert abs(r.hdl_cycles - cycles_exact) <= tol, (
            f"[{r.name}] cycles={r.hdl_cycles}, expected {cycles_exact}±{tol}"
        )
    if cycles_approx is not None:
        ratio = r.hdl_cycles / cycles_approx
        assert 0.8 <= ratio <= 1.4, (
            f"[{r.name}] cycles={r.hdl_cycles}, expected ~{cycles_approx} "
            f"(ratio={ratio:.2f})"
        )
    _report(r)


def _report(r):
    mode = "[float-analytical]" if r.float_path else "[hdl-sim]"
    print(
        f"\n  {mode} {r.name}: "
        f"cycles={r.hdl_cycles:,}  "
        f"sim={r.sim_wall_s:.3f}s  "
        f"tg={r.tg_wall:.4f}s  tg_noopt={r.tg_wall_noopt:.4f}s  "
        f"err={r.max_abs_error:.4g}"
    )


# ===========================================================================
# Test 1 — Scalar add (single element)
# Baseline: every FSM overhead with the minimum possible compute.
# Expected: scalar kernel → ~2 cycles.
# ===========================================================================

def test_perf_01_scalar_add():
    """Scalar int32 add — baseline FSM overhead."""
    a = np.array([7], dtype=np.int32)
    b = np.array([5], dtype=np.int32)

    r = run_bench("scalar_add", lambda t: t[0] + t[1], [a, b])
    _check(r)
    # Scalar kernel: IDLE → SCALAR → IDLE = ~2 cycles
    assert r.hdl_cycles <= 5, f"Scalar should be <5 cycles, got {r.hdl_cycles}"


# ===========================================================================
# Test 2 — Elementwise ReLU, n=32 (memory-bandwidth bound)
# Expected: ~33 cycles (1 loop of 32 + 1 done).
# ===========================================================================

def test_perf_02_elementwise_relu_32():
    """Elementwise ReLU over 32 int32 elements."""
    N = 32
    a = rng.randint(-20, 20, N).astype(np.int32)

    r = run_bench("relu_32", lambda t: t[0].relu(), [a])
    _check(r, cycles_approx=N + 1)


# ===========================================================================
# Test 3 — Elementwise add + ReLU, n=128
# Compound single-kernel elementwise.  Tests larger memory footprint.
# ===========================================================================

def test_perf_03_elementwise_add_relu_128():
    """Elementwise add + relu over 128 int32 elements."""
    N = 128
    a = rng.randint(-10, 10, N).astype(np.int32)
    b = rng.randint(-10, 10, N).astype(np.int32)

    r = run_bench("add_relu_128", lambda t: (t[0] + t[1]).relu(), [a, b])
    _check(r, cycles_approx=N + 1)


# ===========================================================================
# Test 4 — Tiny GEMV: (1, 4) @ (4, 8), int8 → int32
# Compute-bound, smallest non-trivial matmul.
# Expected cycles: 8 * (4 + 2) + 1 = 49.
# ===========================================================================

def test_perf_04_gemv_4x8_int8():
    """Tiny GEMV (1,4)@(4,8) — int8 weights, int32 output."""
    M, K = 8, 4
    x = rng.randint(-4, 4, (1, K)).astype(np.int8)
    w = rng.randint(-4, 4, (K, M)).astype(np.int8)

    r = run_bench(
        "gemv_K4_M8",
        lambda t: (t[0] @ t[1]).cast(dtypes.int32),
        [x, w],
    )
    _check(r, cycles_exact=_expected_gemv_cycles(M, K))


# ===========================================================================
# Test 5 — Small GEMV: (1, 8) @ (8, 16) + bias, int8 → int32
# Adds bias to test compound single-kernel behaviour.
# Expected cycles: 16 * (8 + 2) + 1 = 161.
# ===========================================================================

def test_perf_05_gemv_bias_8x16_int8():
    """GEMV + bias (1,8)@(8,16) — single compound kernel."""
    M, K = 16, 8
    x = rng.randint(-4, 4, (1, K)).astype(np.int8)
    w = rng.randint(-4, 4, (K, M)).astype(np.int8)
    b = rng.randint(-20, 20, (1, M)).astype(np.int32)

    r = run_bench(
        "gemv_bias_K8_M16",
        lambda t: (t[0] @ t[1]).cast(dtypes.int32) + t[2],
        [x, w, b],
    )
    _check(r, cycles_exact=_expected_gemv_cycles(M, K))


# ===========================================================================
# Test 6 — Linear + Bias + ReLU: (1, 8) @ (8, 16) + bias, relu
# Full MNIST-layer pattern in a single kernel.
# Expected cycles: same as Test 5 (relu is in same FSM body).
# ===========================================================================

def test_perf_06_linear_bias_relu_8x16_int8():
    """Linear + bias + relu in a single kernel — MNIST layer pattern."""
    M, K = 16, 8
    x = rng.randint(-3, 3, (1, K)).astype(np.int8)
    w = rng.randint(-3, 3, (K, M)).astype(np.int8)
    b = rng.randint(-15, 15, (1, M)).astype(np.int32)

    r = run_bench(
        "linear_relu_K8_M16",
        lambda t: ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu(),
        [x, w, b],
    )
    _check(r, cycles_exact=_expected_gemv_cycles(M, K))


# ===========================================================================
# Test 7 — 2-layer MLP, small: (1,4)→(1,4)→(1,2)
# First multi-kernel test.  Two chained GEMV kernels.
# ===========================================================================

def test_perf_07_mlp_2layer_small():
    """Two-kernel MLP (1,4)→(1,4)→(1,2) — baseline multi-kernel."""
    K1, M1, M2 = 4, 4, 2
    x  = rng.randint(-4, 4, (1, K1)).astype(np.int8)
    w1 = rng.randint(-3, 3, (K1, M1)).astype(np.int8)
    b1 = rng.randint(-10, 10, (1, M1)).astype(np.int32)
    w2 = rng.randint(-3, 3, (M1, M2)).astype(np.int8)
    b2 = rng.randint(-5, 5, (1, M2)).astype(np.int32)

    def build(t):
        h = ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu()
        return (h.cast(dtypes.int8) @ t[3]).cast(dtypes.int32) + t[4]

    r = run_bench("mlp_2layer_4_4_2", build, [x, w1, b1, w2, b2])
    expected = _expected_gemv_cycles(M1, K1) + _expected_gemv_cycles(M2, M1)
    _check(r, cycles_approx=expected)


# ===========================================================================
# Test 8 — 2-layer MLP, medium: (1,16)→(1,16)→(1,8)
# Larger multi-kernel test to measure scaling.
# ===========================================================================

def test_perf_08_mlp_2layer_medium():
    """Two-kernel MLP (1,16)→(1,16)→(1,8) — medium multi-kernel."""
    K1, M1, M2 = 16, 16, 8
    x  = rng.randint(-4, 4, (1, K1)).astype(np.int8)
    w1 = rng.randint(-2, 2, (K1, M1)).astype(np.int8)
    b1 = rng.randint(-10, 10, (1, M1)).astype(np.int32)
    w2 = rng.randint(-2, 2, (M1, M2)).astype(np.int8)
    b2 = rng.randint(-5, 5, (1, M2)).astype(np.int32)

    def build(t):
        h = ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu()
        return (h.cast(dtypes.int8) @ t[3]).cast(dtypes.int32) + t[4]

    r = run_bench("mlp_2layer_16_16_8", build, [x, w1, b1, w2, b2])
    expected = _expected_gemv_cycles(M1, K1) + _expected_gemv_cycles(M2, M1)
    _check(r, cycles_approx=expected)


# ===========================================================================
# Test 9 — Float32 elementwise and GEMV
# Verifies the float analytical path: correct output, cycles reported.
# ===========================================================================

@pytest.mark.parametrize("op_name,build_fn,arrays", [
    pytest.param(
        "relu_float32_16",
        lambda t: t[0].relu(),
        [rng.randn(16).astype(np.float32)],
        id="relu_float32_16",
    ),
    pytest.param(
        "add_float32_32",
        lambda t: t[0] + t[1],
        [rng.randn(32).astype(np.float32), rng.randn(32).astype(np.float32)],
        id="add_float32_32",
    ),
    pytest.param(
        "gemv_float32_4x8",
        lambda t: t[0] @ t[1],
        [
            rng.randn(1, 4).astype(np.float32),
            rng.randn(4, 8).astype(np.float32),
        ],
        id="gemv_float32_4x8",
    ),
])
def test_perf_09_float32_path(op_name, build_fn, arrays):
    """Float32 workloads — full Amaranth hardware simulation with IEEE 754 FP units.

    The harness now runs float32 through real Amaranth simulation using the
    FP32Add / FP32Mul / FP32Cmp modules, giving bit-accurate IEEE 754 results.
    Correctness is verified against tinygrad CPU with rtol=1e-5.
    """
    r = run_bench(op_name, build_fn, arrays)
    assert not r.float_path, "Float32 should now use hardware simulation, not analytical path"
    assert r.correct, f"IEEE 754 simulation should match CPU reference; got {r}"
    assert r.hdl_cycles > 0, "Cycle count must be positive"
    _report(r)


# ===========================================================================
# Test 10 — MNIST-scale GEMV: (1, 784) @ (784, 128) + bias + relu  [SLOW]
# Measures the cycle count and simulation time for a real MNIST layer.
# Expected cycles: 128 * (784 + 2) + 1 = 100,609.
# ===========================================================================

@pytest.mark.slow
def test_perf_10_mnist_layer1():
    """MNIST-scale layer 1: (1,784)@(784,128) + bias + relu [SLOW]."""
    K, M = 784, 128
    x = rng.randint(-10, 10, (1, K)).astype(np.int8)
    w = rng.randint(-2, 2, (K, M)).astype(np.int8)
    b = rng.randint(-50, 50, (1, M)).astype(np.int32)

    r = run_bench(
        "mnist_layer1_784x128",
        lambda t: ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu(),
        [x, w, b],
    )
    expected = _expected_gemv_cycles(M, K)
    _check(r, cycles_exact=expected)
    print(f"\n  MNIST layer1: {r.hdl_cycles:,} cycles "
          f"({r.hdl_cycles * 10 / 1e6:.3f} ms at 100 MHz)")


# ===========================================================================
# Test 11 — REDUCE unroll: GEMV (1,8)@(8,16), factor=2
# Verifies cycle reduction from REDUCE-axis unrolling (2 MACs/cycle).
# Baseline: 16*(8+2)+1=161.  Unrolled: 16*(4+2)+1=97.
# ===========================================================================

def test_perf_11_reduce_unroll_gemv_2x():
    """GEMV (1,8)@(8,16) with REDUCE unroll factor 2 — 2 MACs/cycle."""
    M, K = 16, 8
    x = rng.randint(-4, 4, (1, K)).astype(np.int8)
    w = rng.randint(-4, 4, (K, M)).astype(np.int8)

    baseline = run_bench(
        "gemv_K8_M16_baseline",
        lambda t: (t[0] @ t[1]).cast(dtypes.int32),
        [x, w],
    )
    unrolled = run_bench(
        "gemv_K8_M16_reduce2x",
        lambda t: (t[0] @ t[1]).cast(dtypes.int32),
        [x, w],
        reduce_unroll_factor=2,
    )

    _check(baseline, cycles_exact=_expected_gemv_cycles(M, K))
    _check(unrolled, cycles_exact=_expected_gemv_cycles(M, K // 2))
    assert unrolled.hdl_cycles < baseline.hdl_cycles, (
        f"Unrolled ({unrolled.hdl_cycles}) should be fewer cycles "
        f"than baseline ({baseline.hdl_cycles})"
    )
    print(f"\n  Speedup: {baseline.hdl_cycles / unrolled.hdl_cycles:.2f}x")


# ===========================================================================
# Test 12 — REDUCE unroll: GEMV (1,8)@(8,16), factor=4
# 4 MACs/cycle.  Baseline: 161.  Unrolled: 16*(2+2)+1=65.
# ===========================================================================

def test_perf_12_reduce_unroll_gemv_4x():
    """GEMV (1,8)@(8,16) with REDUCE unroll factor 4 — 4 MACs/cycle."""
    M, K = 16, 8
    x = rng.randint(-4, 4, (1, K)).astype(np.int8)
    w = rng.randint(-4, 4, (K, M)).astype(np.int8)

    baseline = run_bench(
        "gemv_K8_M16_baseline_4x",
        lambda t: (t[0] @ t[1]).cast(dtypes.int32),
        [x, w],
    )
    unrolled = run_bench(
        "gemv_K8_M16_reduce4x",
        lambda t: (t[0] @ t[1]).cast(dtypes.int32),
        [x, w],
        reduce_unroll_factor=4,
    )

    _check(baseline, cycles_exact=_expected_gemv_cycles(M, K))
    _check(unrolled, cycles_exact=_expected_gemv_cycles(M, K // 4))
    speedup = baseline.hdl_cycles / unrolled.hdl_cycles
    assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.2f}x"
    print(f"\n  Speedup: {speedup:.2f}x")


# ===========================================================================
# Test 13 — REDUCE unroll: linear + bias + relu (1,8)@(8,16) + bias, factor=2
# Full MNIST-layer pattern with unrolling.
# ===========================================================================

def test_perf_13_reduce_unroll_linear_relu():
    """Linear + bias + relu with REDUCE unroll factor 2."""
    M, K = 16, 8
    x = rng.randint(-3, 3, (1, K)).astype(np.int8)
    w = rng.randint(-3, 3, (K, M)).astype(np.int8)
    b = rng.randint(-15, 15, (1, M)).astype(np.int32)

    baseline = run_bench(
        "linear_relu_K8_M16_baseline",
        lambda t: ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu(),
        [x, w, b],
    )
    unrolled = run_bench(
        "linear_relu_K8_M16_reduce2x",
        lambda t: ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu(),
        [x, w, b],
        reduce_unroll_factor=2,
    )

    _check(baseline, cycles_exact=_expected_gemv_cycles(M, K))
    _check(unrolled, cycles_exact=_expected_gemv_cycles(M, K // 2))
    assert unrolled.hdl_cycles < baseline.hdl_cycles
    print(f"\n  Speedup: {baseline.hdl_cycles / unrolled.hdl_cycles:.2f}x")


# ===========================================================================
# Test 14 — REDUCE unroll: 2-layer MLP with unrolling
# Multi-kernel test: both kernels get REDUCE unrolled.
# ===========================================================================

def test_perf_14_reduce_unroll_mlp_2layer():
    """Two-kernel MLP (1,4)→(1,4)→(1,2) with REDUCE unroll factor 2."""
    K1, M1, M2 = 4, 4, 2
    x  = rng.randint(-4, 4, (1, K1)).astype(np.int8)
    w1 = rng.randint(-3, 3, (K1, M1)).astype(np.int8)
    b1 = rng.randint(-10, 10, (1, M1)).astype(np.int32)
    w2 = rng.randint(-3, 3, (M1, M2)).astype(np.int8)
    b2 = rng.randint(-5, 5, (1, M2)).astype(np.int32)

    def build(t):
        h = ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu()
        return (h.cast(dtypes.int8) @ t[3]).cast(dtypes.int32) + t[4]

    baseline = run_bench("mlp_2layer_baseline", build, [x, w1, b1, w2, b2])
    unrolled = run_bench(
        "mlp_2layer_reduce2x", build, [x, w1, b1, w2, b2],
        reduce_unroll_factor=2,
    )

    _check(baseline)
    _check(unrolled)
    assert unrolled.hdl_cycles < baseline.hdl_cycles, (
        f"Unrolled ({unrolled.hdl_cycles}) should be fewer than "
        f"baseline ({baseline.hdl_cycles})"
    )
    print(f"\n  Speedup: {baseline.hdl_cycles / unrolled.hdl_cycles:.2f}x")


# ===========================================================================
# Test 15 — LOOP unroll: elementwise ReLU, n=32, factor=4
# Verifies LOOP-axis unrolling produces correct results.
# LOOP unroll doesn't reduce cycles (write-port bottleneck), but must stay
# correct.
# ===========================================================================

def test_perf_15_loop_unroll_relu():
    """Elementwise ReLU n=32 with LOOP unroll factor 4 — correctness check."""
    N = 32
    a = rng.randint(-20, 20, N).astype(np.int32)

    r = run_bench(
        "relu_32_loop4x",
        lambda t: t[0].relu(),
        [a],
        unroll_factor=4,
    )
    _check(r)


# ===========================================================================
# Test 16 — TopModule + REDUCE unroll: 2-layer MLP through compile_top_module
# Exercises the full TopModule path (compile_top_module → simulate_top) with
# unrolling, proving the integration is end-to-end.
# ===========================================================================

def test_perf_16_topmodule_reduce_unroll():
    """2-layer MLP through compile_top_module + simulate_top with REDUCE unroll."""
    from compiler import compile_top_module
    from compiler.top_module import simulate_top

    K1, M1, M2 = 4, 4, 2
    x_np  = rng.randint(-4, 4, (1, K1)).astype(np.int8)
    w1_np = rng.randint(-3, 3, (K1, M1)).astype(np.int8)
    b1_np = rng.randint(-10, 10, (1, M1)).astype(np.int32)
    w2_np = rng.randint(-3, 3, (M1, M2)).astype(np.int8)
    b2_np = rng.randint(-5, 5, (1, M2)).astype(np.int32)

    # --- CPU reference ---
    ref = (
        (
            (Tensor(x_np) @ Tensor(w1_np)).cast(dtypes.int32) + Tensor(b1_np)
        ).relu().cast(dtypes.int8) @ Tensor(w2_np)
    ).cast(dtypes.int32) + Tensor(b2_np)
    ref_out = ref.numpy().flatten()

    # --- Build schedule ---
    x_sym  = Tensor.empty(1, K1, dtype=dtypes.int8)
    w1_sym = Tensor.empty(K1, M1, dtype=dtypes.int8)
    b1_sym = Tensor.empty(1, M1, dtype=dtypes.int32)
    w2_sym = Tensor.empty(M1, M2, dtype=dtypes.int8)
    b2_sym = Tensor.empty(1, M2, dtype=dtypes.int32)

    h = ((x_sym @ w1_sym).cast(dtypes.int32) + b1_sym).relu()
    logits = (h.cast(dtypes.int8) @ w2_sym).cast(dtypes.int32) + b2_sym

    import os
    old_noopt = os.environ.get("NOOPT")
    os.environ["NOOPT"] = "1"
    try:
        schedule = logits.schedule()
    finally:
        if old_noopt is None:
            os.environ.pop("NOOPT", None)
        else:
            os.environ["NOOPT"] = old_noopt

    # --- Baseline (no unroll) via TopModule ---
    top_base, conns, ks_base = compile_top_module(schedule)
    input_data_base = _build_top_input(top_base, ks_base,
                                       [x_np, w1_np, b1_np, w2_np, b2_np])
    out_base, cc_base, _ = simulate_top(top_base, input_data_base)
    cyc_base = cc_base["compute"]

    # --- Unrolled via TopModule ---
    top_unr, conns_unr, ks_unr = compile_top_module(
        schedule, reduce_unroll_factor=2,
    )
    input_data_unr = _build_top_input(top_unr, ks_unr,
                                      [x_np, w1_np, b1_np, w2_np, b2_np])
    out_unr, cc_unr, _ = simulate_top(top_unr, input_data_unr)
    cyc_unr = cc_unr["compute"]

    # Correctness
    np.testing.assert_array_equal(out_base, ref_out,
                                  err_msg="Baseline TopModule output wrong")
    np.testing.assert_array_equal(out_unr, ref_out,
                                  err_msg="Unrolled TopModule output wrong")

    # Cycle improvement
    assert cyc_unr < cyc_base, (
        f"Unrolled TopModule ({cyc_unr} cycles) should be faster than "
        f"baseline ({cyc_base} cycles)"
    )
    print(f"\n  TopModule baseline: {cyc_base} cycles")
    print(f"  TopModule unrolled: {cyc_unr} cycles")
    print(f"  Speedup: {cyc_base / cyc_unr:.2f}x")


def _build_top_input(top, kernel_specs, arrays_flat):
    """Map flat input arrays to (k_idx, buf_idx) for simulate_top.

    Assigns arrays to external write ports in natural schedule order.
    """
    import numpy as np
    input_data = {}
    arr_iter = iter(arrays_flat)
    # Sort ext_write_ports by (k_idx, buf_idx) for deterministic assignment
    for k_idx, buf_idx in sorted(top.ext_write_ports.keys()):
        try:
            arr = next(arr_iter)
            input_data[(k_idx, buf_idx)] = arr.flatten()
        except StopIteration:
            break
    return input_data
