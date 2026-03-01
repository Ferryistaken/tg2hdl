"""Unit tests for IEEE 754 float32 Amaranth hardware modules.

Tests verify that FP32Add, FP32Mul, and FP32Cmp produce bit-accurate
results matching Python's IEEE 754 float arithmetic via Amaranth simulation.
"""

import os
os.environ.setdefault("NOOPT", "1")
os.environ.setdefault("DEBUG", "0")

import struct
import numpy as np
import pytest
from amaranth.sim import Simulator

from compiler.fp32 import FP32Add, FP32Mul, FP32Cmp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def f32_to_bits(f: float) -> int:
    """Pack a Python float as a 32-bit IEEE 754 unsigned integer."""
    return struct.unpack(">I", struct.pack(">f", f))[0]


def bits_to_f32(b: int) -> float:
    """Unpack a 32-bit unsigned integer as a Python float."""
    return struct.unpack(">f", struct.pack(">I", b & 0xFFFFFFFF))[0]


def _sim_comb(dut, inputs: list[tuple], output_names: list[str]) -> dict:
    """Run a purely combinational Amaranth module.

    Parameters
    ----------
    dut          : Elaboratable
    inputs       : list of (signal, int_value) pairs to drive
    output_names : list of attribute names to read from dut

    Returns a dict mapping attribute name → integer read value.
    """
    result = {}

    async def tb(ctx):
        for sig, val in inputs:
            ctx.set(sig, val)
        # delay(0) lets combinational signals propagate in Amaranth 0.5
        await ctx.delay(0)
        for name in output_names:
            result[name] = ctx.get(getattr(dut, name))

    sim = Simulator(dut)
    # if_exists=True: no error when the module has no sync clock domain
    sim.add_clock(1e-8, if_exists=True)
    sim.add_testbench(tb)
    sim.run()
    return result


def sim_fp32add(a: float, b: float) -> float:
    """Simulate FP32Add(a, b) and return the float result."""
    dut = FP32Add(uid=0)
    r = _sim_comb(dut, [(dut.a, f32_to_bits(a)), (dut.b, f32_to_bits(b))], ["result"])
    return bits_to_f32(r["result"])


def sim_fp32mul(a: float, b: float) -> float:
    """Simulate FP32Mul(a, b) and return the float result."""
    dut = FP32Mul(uid=0)
    r = _sim_comb(dut, [(dut.a, f32_to_bits(a)), (dut.b, f32_to_bits(b))], ["result"])
    return bits_to_f32(r["result"])


def sim_fp32cmp(a: float, b: float) -> bool:
    """Simulate FP32Cmp(a, b) → (a < b)."""
    dut = FP32Cmp(uid=0)
    r = _sim_comb(dut, [(dut.a, f32_to_bits(a)), (dut.b, f32_to_bits(b))], ["result"])
    return bool(r["result"])


def close(a: float, b: float, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """Return True if a ≈ b within tolerances."""
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


# ---------------------------------------------------------------------------
# FP32Add tests
# ---------------------------------------------------------------------------

class TestFP32Add:
    def test_positive_add(self):
        assert close(sim_fp32add(1.0, 2.0), 3.0)

    def test_negative_add(self):
        assert close(sim_fp32add(-1.5, -2.5), -4.0)

    def test_mixed_sign(self):
        assert close(sim_fp32add(5.0, -3.0), 2.0)

    def test_cancellation(self):
        """Subtraction that produces zero."""
        assert sim_fp32add(1.0, -1.0) == 0.0

    def test_add_zero(self):
        assert close(sim_fp32add(3.14, 0.0), 3.14)
        assert close(sim_fp32add(0.0, 3.14), 3.14)

    def test_both_zero(self):
        assert sim_fp32add(0.0, 0.0) == 0.0

    def test_large_exponent_diff(self):
        """Small + large: small is effectively 0."""
        result = sim_fp32add(1e10, 1e-10)
        assert close(result, 1e10, rtol=1e-4)

    def test_add_infinity(self):
        inf = float("inf")
        r = sim_fp32add(inf, 1.0)
        assert r == inf

    @pytest.mark.parametrize("a,b", [
        (0.5, 0.5),
        (1.25, 0.75),
        (-3.0, 1.0),
        (100.0, 200.0),
        (0.1, 0.2),
    ])
    def test_parametrized(self, a, b):
        ref = np.float32(a) + np.float32(b)
        got = sim_fp32add(a, b)
        assert close(float(ref), got), f"add({a},{b}): expected {float(ref)}, got {got}"


# ---------------------------------------------------------------------------
# FP32Mul tests
# ---------------------------------------------------------------------------

class TestFP32Mul:
    def test_positive_mul(self):
        assert close(sim_fp32mul(2.0, 3.0), 6.0)

    def test_negative_mul(self):
        assert close(sim_fp32mul(-2.0, 3.0), -6.0)
        assert close(sim_fp32mul(-2.0, -3.0), 6.0)

    def test_mul_zero(self):
        assert sim_fp32mul(1.0, 0.0) == 0.0
        assert sim_fp32mul(0.0, 1.0) == 0.0

    def test_mul_one(self):
        assert close(sim_fp32mul(3.14, 1.0), 3.14)

    def test_mul_infinity(self):
        inf = float("inf")
        r = sim_fp32mul(inf, 2.0)
        assert r == inf

    def test_mul_zero_by_infinity(self):
        inf = float("inf")
        # 0 × inf = 0 (we return 0 when either input is zero)
        r = sim_fp32mul(0.0, inf)
        assert r == 0.0

    @pytest.mark.parametrize("a,b", [
        (1.5, 2.0),
        (0.5, 0.5),
        (-1.0, 4.0),
        (3.0, 3.0),
        (1.25, 1.6),
        (100.0, 0.01),
    ])
    def test_parametrized(self, a, b):
        ref = np.float32(a) * np.float32(b)
        got = sim_fp32mul(a, b)
        assert close(float(ref), got), f"mul({a},{b}): expected {float(ref)}, got {got}"


# ---------------------------------------------------------------------------
# FP32Cmp tests
# ---------------------------------------------------------------------------

class TestFP32Cmp:
    def test_less_than_positive(self):
        assert sim_fp32cmp(1.0, 2.0) is True
        assert sim_fp32cmp(2.0, 1.0) is False

    def test_equal(self):
        assert sim_fp32cmp(1.0, 1.0) is False

    def test_negative_less(self):
        assert sim_fp32cmp(-2.0, -1.0) is True   # -2 < -1
        assert sim_fp32cmp(-1.0, -2.0) is False

    def test_opposite_signs(self):
        assert sim_fp32cmp(-1.0, 1.0) is True    # negative < positive
        assert sim_fp32cmp(1.0, -1.0) is False

    def test_zero_equality(self):
        # -0.0 == +0.0, so neither is less-than
        assert sim_fp32cmp(0.0, -0.0) is False
        assert sim_fp32cmp(-0.0, 0.0) is False

    def test_compare_with_zero(self):
        assert sim_fp32cmp(-1.0, 0.0) is True    # -1 < 0
        assert sim_fp32cmp(1.0, 0.0) is False     # 1 not < 0

    @pytest.mark.parametrize("a,b,expected", [
        (0.5, 1.5, True),
        (1.5, 0.5, False),
        (-3.0, 3.0, True),
        (3.0, -3.0, False),
        (100.0, 100.1, True),
    ])
    def test_parametrized(self, a, b, expected):
        got = sim_fp32cmp(a, b)
        assert got == expected, f"cmp({a} < {b}): expected {expected}, got {got}"


# ---------------------------------------------------------------------------
# Integration: float32 relu via run_bench
# ---------------------------------------------------------------------------

def test_fp32_relu_harness():
    """Verify float32 relu through the full benchmark harness."""
    import numpy as np
    from benchmarks.harness import run_bench

    rng = np.random.RandomState(99)
    a = rng.randn(16).astype(np.float32)
    r = run_bench("relu_fp32_hw", lambda t: t[0].relu(), [a])

    assert not r.float_path, "Should use hardware simulation, not analytical path"
    assert r.correct, f"relu float32 incorrect: {r}"
    assert r.hdl_cycles > 0


def test_fp32_add_harness():
    """Verify float32 elementwise add through the full benchmark harness."""
    import numpy as np
    from benchmarks.harness import run_bench

    rng = np.random.RandomState(77)
    a = rng.randn(8).astype(np.float32)
    b = rng.randn(8).astype(np.float32)
    r = run_bench("add_fp32_hw", lambda t: t[0] + t[1], [a, b])

    assert not r.float_path
    assert r.correct, f"add float32 incorrect: {r}"
