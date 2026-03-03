"""Tests for all 'easy' UOps added in the second lowering pass.

Tests are grouped by which UOp actually appears in the linearized kernel.
Where tinygrad rewrites an op (e.g. SUB → MUL+ADD), the test validates the
end-to-end result through the compiler even though the specific UOp isn't
emitted; this exercises the full pipeline and documents the rewrite.

UOps verified to actually appear in linearized kernels (from UOp probe):
  AND, OR, XOR, CMPNE, IDIV, BITCAST

UOps added for completeness / direct IR use (tinygrad rewrites them away):
  SUB, NEG, SHL, SHR, MOD, CMPEQ, TRUNC
"""

import os

os.environ["NOOPT"] = "1"
os.environ["DEBUG"] = "0"

import numpy as np
import pytest
from tinygrad import Tensor, dtypes

from compiler import HDLRenderer, compile_kernel, simulate_kernel
from compiler.backend import _get_uops
from tinygrad.uop.ops import Ops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compile_and_run(expr_tensor, input_map: dict) -> np.ndarray:
    renderer = HDLRenderer()
    sched = expr_tensor.schedule()
    compute = [si for si in sched if si.ast.op == Ops.SINK]
    assert len(compute) >= 1
    uops = _get_uops(compute[-1].ast, renderer)
    kernel = compile_kernel(uops)
    out, _, _ = simulate_kernel(kernel, input_map)
    return out


def _emitted_ops(expr_tensor) -> list[str]:
    """Return the non-boilerplate UOp names emitted for this expression."""
    renderer = HDLRenderer()
    sched = expr_tensor.schedule()
    for si in sched:
        if si.ast.op == Ops.SINK:
            uops = _get_uops(si.ast, renderer)
            skip = {'DEFINE_GLOBAL', 'CONST', 'RANGE', 'END', 'AFTER',
                    'SINK', 'INDEX', 'LOAD', 'STORE'}
            return [u.op.name for u in uops if u.op.name not in skip]
    return []


# ---------------------------------------------------------------------------
# AND
# ---------------------------------------------------------------------------

def test_and_emits_and_uop():
    a = Tensor.empty(4, dtype=dtypes.int32)
    b = Tensor.empty(4, dtype=dtypes.int32)
    assert "AND" in _emitted_ops(a & b)


def test_and_int32():
    a = np.array([0b1100, 0b1010, 0xFF, 0x00], dtype=np.int32)
    b = np.array([0b1010, 0b0110, 0x0F, 0xFF], dtype=np.int32)
    out = _compile_and_run(
        Tensor.empty(4, dtype=dtypes.int32) & Tensor.empty(4, dtype=dtypes.int32),
        {1: a, 2: b},
    )
    np.testing.assert_array_equal(out, a & b)


def test_and_masks_bits():
    a = np.array([0xFFFFFFFF], dtype=np.uint32).view(np.int32)
    b = np.array([0x0000FFFF], dtype=np.uint32).view(np.int32)
    out = _compile_and_run(
        Tensor.empty(1, dtype=dtypes.int32) & Tensor.empty(1, dtype=dtypes.int32),
        {1: a, 2: b},
    )
    np.testing.assert_array_equal(out.astype(np.uint32), np.array([0x0000FFFF], dtype=np.uint32))


# ---------------------------------------------------------------------------
# OR
# ---------------------------------------------------------------------------

def test_or_emits_or_uop():
    a = Tensor.empty(4, dtype=dtypes.int32)
    b = Tensor.empty(4, dtype=dtypes.int32)
    assert "OR" in _emitted_ops(a | b)


def test_or_int32():
    a = np.array([0b1100, 0b1010, 0x00, 0xF0], dtype=np.int32)
    b = np.array([0b0011, 0b0101, 0xFF, 0x0F], dtype=np.int32)
    out = _compile_and_run(
        Tensor.empty(4, dtype=dtypes.int32) | Tensor.empty(4, dtype=dtypes.int32),
        {1: a, 2: b},
    )
    np.testing.assert_array_equal(out, a | b)


# ---------------------------------------------------------------------------
# XOR
# ---------------------------------------------------------------------------

def test_xor_emits_xor_uop():
    a = Tensor.empty(4, dtype=dtypes.int32)
    b = Tensor.empty(4, dtype=dtypes.int32)
    assert "XOR" in _emitted_ops(a ^ b)


def test_xor_int32():
    a = np.array([0b1111, 0b1010, 0xFF, 5], dtype=np.int32)
    b = np.array([0b1010, 0b1010, 0x0F, 5], dtype=np.int32)
    out = _compile_and_run(
        Tensor.empty(4, dtype=dtypes.int32) ^ Tensor.empty(4, dtype=dtypes.int32),
        {1: a, 2: b},
    )
    np.testing.assert_array_equal(out, a ^ b)


def test_xor_self_is_zero():
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    out = _compile_and_run(
        Tensor.empty(4, dtype=dtypes.int32) ^ Tensor.empty(4, dtype=dtypes.int32),
        {1: a, 2: a},
    )
    np.testing.assert_array_equal(out, np.zeros(4, dtype=np.int32))


# ---------------------------------------------------------------------------
# CMPNE  (== also uses CMPNE internally via rewrite)
# ---------------------------------------------------------------------------

def test_cmpne_emits_cmpne_uop():
    a = Tensor.empty(4, dtype=dtypes.int32)
    b = Tensor.empty(4, dtype=dtypes.int32)
    assert "CMPNE" in _emitted_ops((a != b).cast(dtypes.int32))


def test_cmpne_int32():
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([1, 0, 3, 9], dtype=np.int32)
    out = _compile_and_run(
        (Tensor.empty(4, dtype=dtypes.int32) != Tensor.empty(4, dtype=dtypes.int32)).cast(dtypes.int32),
        {1: a, 2: b},
    )
    expected = (a != b).astype(np.int32)
    np.testing.assert_array_equal(out, expected)


def test_cmpeq_via_cmpne_rewrite():
    """tinygrad implements == as NOT(!=), emitting two CMPNE ops."""
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([1, 0, 3, 9], dtype=np.int32)
    out = _compile_and_run(
        (Tensor.empty(4, dtype=dtypes.int32) == Tensor.empty(4, dtype=dtypes.int32)).cast(dtypes.int32),
        {1: a, 2: b},
    )
    expected = (a == b).astype(np.int32)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# IDIV  (also covers SHR since tinygrad emits IDIV for >> by constant)
# ---------------------------------------------------------------------------

def test_idiv_emits_idiv_uop():
    a = Tensor.empty(4, dtype=dtypes.uint32)
    b = Tensor.empty(4, dtype=dtypes.uint32)
    assert "IDIV" in _emitted_ops(a // b)


def test_idiv_uint32():
    a = np.array([10, 20, 100, 7], dtype=np.uint32)
    b = np.array([2,   3,   7,  3], dtype=np.uint32)
    out = _compile_and_run(
        Tensor.empty(4, dtype=dtypes.uint32) // Tensor.empty(4, dtype=dtypes.uint32),
        {1: a, 2: b},
    )
    np.testing.assert_array_equal(out.astype(np.uint32), a // b)


def test_shr_via_idiv_rewrite():
    """tinygrad compiles >> by constant to IDIV(2^n)."""
    a = np.array([8, 16, 32, 64], dtype=np.uint32)
    out = _compile_and_run(
        Tensor.empty(4, dtype=dtypes.uint32).rshift(2),
        {1: a},
    )
    np.testing.assert_array_equal(out.astype(np.uint32), a >> 2)


# ---------------------------------------------------------------------------
# BITCAST
# ---------------------------------------------------------------------------

def test_bitcast_emits_bitcast_uop():
    a = Tensor.empty(4, dtype=dtypes.int32)
    assert "BITCAST" in _emitted_ops(a.bitcast(dtypes.float32))


def test_bitcast_int32_to_float32_preserves_bits():
    """Bitcast reinterprets bit pattern — no value conversion."""
    import struct
    # Pack known float values as int32 bit patterns
    floats = [1.0, -1.0, 0.5, 3.14]
    bits = np.array([struct.unpack('>I', struct.pack('>f', f))[0] for f in floats],
                    dtype=np.uint32).view(np.int32)
    out = _compile_and_run(
        Tensor.empty(4, dtype=dtypes.int32).bitcast(dtypes.float32),
        {1: bits},
    )
    # Output is returned as int32 bit patterns; reinterpret as float32
    recovered = out.astype(np.uint32).view(np.float32)
    np.testing.assert_array_almost_equal(recovered, np.array(floats, dtype=np.float32))


# ---------------------------------------------------------------------------
# SUB  (tinygrad rewrites to MUL+ADD, but pipeline must handle it correctly)
# ---------------------------------------------------------------------------

def test_sub_result_correct():
    """a - b should compile and produce correct results even via MUL+ADD rewrite."""
    a = np.array([10, 5, 0, -3], dtype=np.int32)
    b = np.array([3,  5, 7,  2], dtype=np.int32)
    out = _compile_and_run(
        Tensor.empty(4, dtype=dtypes.int32) - Tensor.empty(4, dtype=dtypes.int32),
        {1: a, 2: b},
    )
    np.testing.assert_array_equal(out, a - b)


# ---------------------------------------------------------------------------
# NEG  (tinygrad rewrites to MUL(-1))
# ---------------------------------------------------------------------------

def test_neg_result_correct():
    a = np.array([1, -2, 0, 100], dtype=np.int32)
    out = _compile_and_run(
        -Tensor.empty(4, dtype=dtypes.int32),
        {1: a},
    )
    np.testing.assert_array_equal(out, -a)


# ---------------------------------------------------------------------------
# MOD  (tinygrad rewrites to IDIV+MUL+MUL+ADD)
# ---------------------------------------------------------------------------

def test_mod_result_correct():
    a = np.array([10, 17, 100, 7], dtype=np.uint32)
    b = np.array([3,   5,   7,  3], dtype=np.uint32)
    out = _compile_and_run(
        Tensor.empty(4, dtype=dtypes.uint32) % Tensor.empty(4, dtype=dtypes.uint32),
        {1: a, 2: b},
    )
    np.testing.assert_array_equal(out.astype(np.uint32), a % b)
