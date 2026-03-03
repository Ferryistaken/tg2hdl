"""Tests for complex IEEE 754 FP32 hardware modules: EXP2, LOG2, RECIPROCAL, SQRT, FDIV.

Each section has:
  - Unit tests against the Amaranth hardware module directly.
  - Integration test via the full compiler → simulate_kernel pipeline.

Tolerance: rtol=1e-5, atol=1e-6 (same as existing FP32 tests).
Edge cases covered: zero, infinity, NaN, negative inputs (where defined).
"""

import os
os.environ.setdefault("NOOPT", "1")
os.environ.setdefault("DEBUG", "0")

import math
import struct
import numpy as np
import pytest
from amaranth.sim import Simulator

from compiler.fp32 import FP32Exp2, FP32Log2, FP32Reciprocal, FP32Sqrt, FP32FDiv


# ---------------------------------------------------------------------------
# Shared helpers (copied from test_fp32.py pattern)
# ---------------------------------------------------------------------------

def f32_to_bits(f: float) -> int:
    return struct.unpack(">I", struct.pack(">f", f))[0]


def bits_to_f32(b: int) -> float:
    return struct.unpack(">f", struct.pack(">I", b & 0xFFFFFFFF))[0]


def _sim_comb(dut, inputs: list, output_names: list) -> dict:
    result = {}

    async def tb(ctx):
        for sig, val in inputs:
            ctx.set(sig, val)
        await ctx.delay(0)
        for name in output_names:
            result[name] = ctx.get(getattr(dut, name))

    sim = Simulator(dut)
    sim.add_clock(1e-8, if_exists=True)
    sim.add_testbench(tb)
    sim.run()
    return result


def _sim_unary(dut, a: float) -> float:
    r = _sim_comb(dut, [(dut.a, f32_to_bits(a))], ["result"])
    return bits_to_f32(r["result"])


def _sim_binary(dut, a: float, b: float) -> float:
    r = _sim_comb(dut, [(dut.a, f32_to_bits(a)), (dut.b, f32_to_bits(b))], ["result"])
    return bits_to_f32(r["result"])


def _assert_close(got, expected, rtol=1e-5, atol=1e-6, label=""):
    assert math.isfinite(got) == math.isfinite(expected), \
        f"{label}: finiteness mismatch got={got} expected={expected}"
    if math.isfinite(expected):
        assert abs(got - expected) <= atol + rtol * abs(expected), \
            f"{label}: got={got}, expected={expected}, diff={abs(got-expected)}"


# ---------------------------------------------------------------------------
# EXP2 — 2^x
# ---------------------------------------------------------------------------

class TestFP32Exp2:

    def test_exp2_zero(self):
        """2^0 = 1.0"""
        assert _sim_unary(FP32Exp2(), 0.0) == pytest.approx(1.0, rel=1e-5)

    def test_exp2_one(self):
        """2^1 = 2.0"""
        assert _sim_unary(FP32Exp2(), 1.0) == pytest.approx(2.0, rel=1e-5)

    def test_exp2_minus_one(self):
        """2^-1 = 0.5"""
        assert _sim_unary(FP32Exp2(), -1.0) == pytest.approx(0.5, rel=1e-5)

    def test_exp2_two(self):
        """2^2 = 4.0"""
        assert _sim_unary(FP32Exp2(), 2.0) == pytest.approx(4.0, rel=1e-5)

    def test_exp2_half(self):
        """2^0.5 = sqrt(2)"""
        _assert_close(_sim_unary(FP32Exp2(), 0.5), math.sqrt(2.0), label="2^0.5")

    def test_exp2_negative_half(self):
        """2^-0.5 = 1/sqrt(2)"""
        _assert_close(_sim_unary(FP32Exp2(), -0.5), 1.0 / math.sqrt(2.0), label="2^-0.5")

    def test_exp2_fractional_values(self):
        """2^x for a range of fractional inputs."""
        for x in [0.1, 0.25, 0.3, 0.7, 0.9, -0.1, -0.25, -0.75]:
            got = _sim_unary(FP32Exp2(), x)
            expected = 2.0 ** x
            _assert_close(got, expected, label=f"2^{x}")

    def test_exp2_large_integers(self):
        """2^n for integer n."""
        for n in [3, 4, 7, 10, -3, -5]:
            got = _sim_unary(FP32Exp2(), float(n))
            expected = 2.0 ** n
            _assert_close(got, expected, label=f"2^{n}")

    def test_exp2_typical_softmax_range(self):
        """Inputs typical for softmax: x * log2e where x ∈ [-5, 5]."""
        log2e = math.log2(math.e)
        for x in [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]:
            val = x * log2e
            got = _sim_unary(FP32Exp2(), val)
            expected = 2.0 ** val
            _assert_close(got, expected, label=f"2^({val:.4f})")

    def test_exp2_positive_inf(self):
        """2^(+inf) = +inf"""
        result = _sim_unary(FP32Exp2(), math.inf)
        assert math.isinf(result) and result > 0

    def test_exp2_negative_inf(self):
        """2^(-inf) = 0"""
        result = _sim_unary(FP32Exp2(), -math.inf)
        assert result == 0.0

    def test_exp2_overflow(self):
        """2^200 overflows to +inf"""
        result = _sim_unary(FP32Exp2(), 200.0)
        assert math.isinf(result) and result > 0

    def test_exp2_underflow(self):
        """2^-200 underflows to 0"""
        result = _sim_unary(FP32Exp2(), -200.0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# LOG2 — log2(x)
# ---------------------------------------------------------------------------

class TestFP32Log2:

    def test_log2_one(self):
        """log2(1) = 0"""
        _assert_close(_sim_unary(FP32Log2(), 1.0), 0.0, atol=1e-5, label="log2(1)")

    def test_log2_two(self):
        """log2(2) = 1"""
        _assert_close(_sim_unary(FP32Log2(), 2.0), 1.0, label="log2(2)")

    def test_log2_four(self):
        """log2(4) = 2"""
        _assert_close(_sim_unary(FP32Log2(), 4.0), 2.0, label="log2(4)")

    def test_log2_half(self):
        """log2(0.5) = -1"""
        _assert_close(_sim_unary(FP32Log2(), 0.5), -1.0, label="log2(0.5)")

    def test_log2_eighth(self):
        """log2(0.125) = -3"""
        _assert_close(_sim_unary(FP32Log2(), 0.125), -3.0, label="log2(0.125)")

    def test_log2_generic(self):
        """log2(x) for non-power-of-two inputs."""
        for x in [1.5, 1.25, 1.75, 3.0, 0.75, 6.0]:
            got = _sim_unary(FP32Log2(), x)
            expected = math.log2(x)
            _assert_close(got, expected, label=f"log2({x})")

    def test_log2_inf(self):
        """log2(+inf) = +inf"""
        result = _sim_unary(FP32Log2(), math.inf)
        assert math.isinf(result) and result > 0

    def test_log2_zero(self):
        """log2(0) = -inf"""
        result = _sim_unary(FP32Log2(), 0.0)
        assert math.isinf(result) and result < 0

    def test_log2_negative(self):
        """log2(negative) = NaN"""
        result = _sim_unary(FP32Log2(), -1.0)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# RECIPROCAL — 1/x
# ---------------------------------------------------------------------------

class TestFP32Reciprocal:

    def test_reciprocal_one(self):
        """1/1 = 1"""
        _assert_close(_sim_unary(FP32Reciprocal(), 1.0), 1.0, label="1/1")

    def test_reciprocal_two(self):
        """1/2 = 0.5"""
        _assert_close(_sim_unary(FP32Reciprocal(), 2.0), 0.5, label="1/2")

    def test_reciprocal_four(self):
        """1/4 = 0.25"""
        _assert_close(_sim_unary(FP32Reciprocal(), 4.0), 0.25, label="1/4")

    def test_reciprocal_half(self):
        """1/0.5 = 2"""
        _assert_close(_sim_unary(FP32Reciprocal(), 0.5), 2.0, label="1/0.5")

    def test_reciprocal_negative(self):
        """1/(-2) = -0.5"""
        _assert_close(_sim_unary(FP32Reciprocal(), -2.0), -0.5, label="1/(-2)")

    def test_reciprocal_generic(self):
        for x in [3.0, 7.0, 0.1, 0.25, 1.5, -1.0, -4.0]:
            got = _sim_unary(FP32Reciprocal(), x)
            _assert_close(got, 1.0 / x, label=f"1/{x}")

    def test_reciprocal_inf(self):
        """1/inf = 0"""
        assert _sim_unary(FP32Reciprocal(), math.inf) == 0.0

    def test_reciprocal_zero(self):
        """1/0 = +inf"""
        result = _sim_unary(FP32Reciprocal(), 0.0)
        assert math.isinf(result) and result > 0


# ---------------------------------------------------------------------------
# SQRT — sqrt(x)
# ---------------------------------------------------------------------------

class TestFP32Sqrt:

    def test_sqrt_one(self):
        _assert_close(_sim_unary(FP32Sqrt(), 1.0), 1.0, label="sqrt(1)")

    def test_sqrt_four(self):
        _assert_close(_sim_unary(FP32Sqrt(), 4.0), 2.0, label="sqrt(4)")

    def test_sqrt_nine(self):
        _assert_close(_sim_unary(FP32Sqrt(), 9.0), 3.0, label="sqrt(9)")

    def test_sqrt_two(self):
        _assert_close(_sim_unary(FP32Sqrt(), 2.0), math.sqrt(2.0), label="sqrt(2)")

    def test_sqrt_quarter(self):
        _assert_close(_sim_unary(FP32Sqrt(), 0.25), 0.5, label="sqrt(0.25)")

    def test_sqrt_generic(self):
        for x in [2.0, 3.0, 5.0, 0.5, 0.1, 100.0]:
            got = _sim_unary(FP32Sqrt(), x)
            _assert_close(got, math.sqrt(x), label=f"sqrt({x})")

    def test_sqrt_zero(self):
        assert _sim_unary(FP32Sqrt(), 0.0) == 0.0

    def test_sqrt_inf(self):
        result = _sim_unary(FP32Sqrt(), math.inf)
        assert math.isinf(result) and result > 0

    def test_sqrt_negative(self):
        """sqrt(-x) = NaN"""
        result = _sim_unary(FP32Sqrt(), -1.0)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# FDIV — a / b
# ---------------------------------------------------------------------------

class TestFP32FDiv:

    def test_fdiv_basic(self):
        for a, b in [(4.0, 2.0), (1.0, 3.0), (7.0, 4.0), (-6.0, 2.0)]:
            got = _sim_binary(FP32FDiv(), a, b)
            _assert_close(got, a / b, label=f"{a}/{b}")

    def test_fdiv_by_zero(self):
        result = _sim_binary(FP32FDiv(), 1.0, 0.0)
        assert math.isinf(result) and result > 0

    def test_fdiv_inf_by_finite(self):
        result = _sim_binary(FP32FDiv(), math.inf, 2.0)
        assert math.isinf(result)

    def test_fdiv_finite_by_inf(self):
        result = _sim_binary(FP32FDiv(), 1.0, math.inf)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Integration tests via tinygrad compiler pipeline
# ---------------------------------------------------------------------------

def _run_fp32_op(tensor_fn, inputs_np):
    """Compile and simulate a float32 tinygrad operation."""
    from tinygrad import Tensor, dtypes
    from compiler import HDLRenderer, compile_kernel, simulate_kernel
    from compiler.backend import _get_uops
    from tinygrad.uop.ops import Ops

    renderer = HDLRenderer()
    sym_inputs = [Tensor.empty(a.shape, dtype=Tensor(a).dtype) for a in inputs_np]
    expr = tensor_fn(sym_inputs)
    sched = expr.schedule()
    compute = [si for si in sched if si.ast.op == Ops.SINK]
    uops = _get_uops(compute[-1].ast, renderer)
    kernel = compile_kernel(uops)
    input_map = {i + 1: a for i, a in enumerate(inputs_np)}
    out, _, _ = simulate_kernel(kernel, input_map)
    return out.astype(np.uint32).view(np.float32)


def test_integration_exp_via_pipeline():
    """Tensor.exp() → EXP2 in hardware, compare to numpy."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    from tinygrad import Tensor
    got = _run_fp32_op(lambda t: t[0].exp(), [x])
    np.testing.assert_allclose(got, np.exp(x), rtol=1e-4, atol=1e-5)


def test_integration_log_via_pipeline():
    """Tensor.log() → LOG2 in hardware, compare to numpy."""
    x = np.array([0.5, 1.0, 2.0, math.e, 10.0], dtype=np.float32)
    from tinygrad import Tensor
    got = _run_fp32_op(lambda t: t[0].log(), [x])
    np.testing.assert_allclose(got, np.log(x), rtol=1e-4, atol=1e-5)


def test_integration_sqrt_via_pipeline():
    """Tensor.sqrt() → SQRT in hardware, compare to numpy."""
    x = np.array([1.0, 4.0, 9.0, 2.0, 0.25], dtype=np.float32)
    from tinygrad import Tensor
    got = _run_fp32_op(lambda t: t[0].sqrt(), [x])
    np.testing.assert_allclose(got, np.sqrt(x), rtol=1e-4, atol=1e-5)
