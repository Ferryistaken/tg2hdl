"""Pytest benchmark suite.

Three tiers of correctness tests comparing tinygrad CPU to HDL simulation:

  Tier 1 — Elementwise (fast, exact int32)
  Tier 2 — Single-kernel GEMV (exact with int8 truncation semantics)
  Tier 3 — Multi-kernel chained (2-layer MLP)

Run with:
    uv run pytest benchmarks/suite.py -v

Each test asserts result.correct and prints cycle counts.
"""

import os

os.environ.setdefault("NOOPT", "1")
os.environ.setdefault("DEBUG", "0")

import numpy as np
import pytest
from tinygrad import Tensor, dtypes

from benchmarks.harness import run_bench

# Deterministic RNG
_rng = np.random.RandomState(7)


# ---------------------------------------------------------------------------
# Tier 1 — Elementwise (exact int32)
# ---------------------------------------------------------------------------

def _t1_add():
    a = _rng.randint(-50, 50, 4).astype(np.int32)
    b = _rng.randint(-50, 50, 4).astype(np.int32)
    return "add_4", lambda t: t[0] + t[1], [a, b]


def _t1_relu():
    a = _rng.randint(-20, 20, 6).astype(np.int32)
    return "relu_6", lambda t: t[0].relu(), [a]


def _t1_relu_add_const():
    a = _rng.randint(-10, 10, 5).astype(np.int32)
    b = _rng.randint(-10, 10, 5).astype(np.int32)
    return (
        "relu_add_const_5",
        lambda t: (t[0] + t[1] + 3).relu(),
        [a, b],
    )


# ---------------------------------------------------------------------------
# Tier 2 — Single-kernel GEMV (int8 weights, int32 output)
# ---------------------------------------------------------------------------

def _t2_matmul():
    x = _rng.randint(-5, 5, (1, 4)).astype(np.int8)
    w = _rng.randint(-5, 5, (4, 8)).astype(np.int8)
    return (
        "matmul_1x4_4x8",
        lambda t: (t[0] @ t[1]).cast(dtypes.int32),
        [x, w],
    )


def _t2_matmul_bias():
    x = _rng.randint(-5, 5, (1, 4)).astype(np.int8)
    w = _rng.randint(-5, 5, (4, 8)).astype(np.int8)
    b = _rng.randint(-20, 20, (1, 8)).astype(np.int32)
    return (
        "matmul_bias_1x4_4x8",
        lambda t: (t[0] @ t[1]).cast(dtypes.int32) + t[2],
        [x, w, b],
    )


def _t2_matmul_bias_relu():
    x = _rng.randint(-3, 3, (1, 3)).astype(np.int8)
    w = _rng.randint(-3, 3, (3, 5)).astype(np.int8)
    b = _rng.randint(-15, 15, (1, 5)).astype(np.int32)
    return (
        "matmul_bias_relu_1x3_3x5",
        lambda t: ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu(),
        [x, w, b],
    )


# ---------------------------------------------------------------------------
# Tier 3 — Multi-kernel (2-layer MLP)
# ---------------------------------------------------------------------------

def _t3_mlp_small():
    x = _rng.randint(-4, 4, (1, 4)).astype(np.int8)
    w1 = _rng.randint(-3, 3, (4, 3)).astype(np.int8)
    b1 = _rng.randint(-10, 10, (1, 3)).astype(np.int32)
    w2 = _rng.randint(-3, 3, (3, 2)).astype(np.int8)
    b2 = _rng.randint(-5, 5, (1, 2)).astype(np.int32)

    def build(t):
        h = ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu()
        return (h.cast(dtypes.int8) @ t[3]).cast(dtypes.int32) + t[4]

    return "mlp_4_3_2", build, [x, w1, b1, w2, b2]


def _t3_mlp_medium():
    x = _rng.randint(-4, 4, (1, 8)).astype(np.int8)
    w1 = _rng.randint(-3, 3, (8, 4)).astype(np.int8)
    b1 = _rng.randint(-10, 10, (1, 4)).astype(np.int32)
    w2 = _rng.randint(-3, 3, (4, 3)).astype(np.int8)
    b2 = _rng.randint(-5, 5, (1, 3)).astype(np.int32)

    def build(t):
        h = ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu()
        return (h.cast(dtypes.int8) @ t[3]).cast(dtypes.int32) + t[4]

    return "mlp_8_4_3", build, [x, w1, b1, w2, b2]


# ---------------------------------------------------------------------------
# Parametrize
# ---------------------------------------------------------------------------

_TIER1 = [_t1_add, _t1_relu, _t1_relu_add_const]
_TIER2 = [_t2_matmul, _t2_matmul_bias, _t2_matmul_bias_relu]
_TIER3 = [_t3_mlp_small, _t3_mlp_medium]


def _make_params(factories, exact=True):
    params = []
    for factory in factories:
        name, build_fn, arrays = factory()
        params.append(pytest.param(name, build_fn, arrays, exact, id=name))
    return params


@pytest.mark.parametrize("name,build_fn,arrays,exact", _make_params(_TIER1))
def test_tier1_elementwise(name, build_fn, arrays, exact):
    r = run_bench(name, build_fn, arrays, exact=exact)
    print(f"\n  {r}")
    assert r.correct, (
        f"{name}: max_abs_error={r.max_abs_error}\n"
        f"  ref={r.output_ref}\n  hdl={r.output_hdl}"
    )


@pytest.mark.parametrize("name,build_fn,arrays,exact", _make_params(_TIER2))
def test_tier2_single_kernel_gemv(name, build_fn, arrays, exact):
    r = run_bench(name, build_fn, arrays, exact=exact)
    print(f"\n  {r}")
    assert r.correct, (
        f"{name}: max_abs_error={r.max_abs_error}\n"
        f"  ref={r.output_ref}\n  hdl={r.output_hdl}"
    )


@pytest.mark.parametrize("name,build_fn,arrays,exact", _make_params(_TIER3))
def test_tier3_multi_kernel(name, build_fn, arrays, exact):
    r = run_bench(name, build_fn, arrays, exact=exact)
    print(f"\n  {r}")
    assert r.correct, (
        f"{name}: max_abs_error={r.max_abs_error}\n"
        f"  ref={r.output_ref}\n  hdl={r.output_hdl}"
    )
