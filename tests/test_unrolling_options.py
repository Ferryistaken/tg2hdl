"""Tests for LOOP unroll options/guardrails plumbing."""

import os

os.environ["NOOPT"] = "1"
os.environ["DEBUG"] = "0"

import numpy as np
from tinygrad import Tensor, dtypes

from compiler import HDLRenderer
from compiler.backend import _get_uops, compile_kernel, simulate_kernel


def _build_add_uops(n: int = 8):
    renderer = HDLRenderer()
    a = Tensor.empty(n, dtype=dtypes.int32)
    b = Tensor.empty(n, dtype=dtypes.int32)
    out = a + b
    sched = out.schedule()
    return _get_uops(sched[0].ast, renderer)


def test_unroll_default_report():
    uops = _build_add_uops(4)
    kernel = compile_kernel(uops)

    a_data = np.array([1, 2, 3, 4], dtype=np.int32)
    b_data = np.array([10, 20, 30, 40], dtype=np.int32)
    output, _, _ = simulate_kernel(kernel, {1: a_data, 2: b_data})

    np.testing.assert_array_equal(output, a_data + b_data)
    assert kernel.compile_report["unroll_loop_requested"] == 1
    assert kernel.compile_report["unroll_loop_applied"] == 1
    assert kernel.compile_report["unroll_fallback_reason"] is None


def test_unroll_request_falls_back_cleanly():
    uops = _build_add_uops(10)
    kernel = compile_kernel(uops, unroll_loop=4)

    a_data = np.arange(10, dtype=np.int32)
    b_data = (np.arange(10, dtype=np.int32) * 2) - 3
    output, _, _ = simulate_kernel(kernel, {1: a_data, 2: b_data})

    np.testing.assert_array_equal(output, a_data + b_data)
    assert kernel.compile_report["unroll_loop_requested"] == 4
    assert kernel.compile_report["unroll_loop_applied"] == 1
    assert kernel.compile_report["unroll_fallback_reason"] is not None


def test_unroll_loop_reduces_cycles_for_elementwise_kernel():
    uops = _build_add_uops(16)
    a_data = np.arange(16, dtype=np.int32)
    b_data = (np.arange(16, dtype=np.int32) * 5) - 7

    base = compile_kernel(uops, unroll_loop=1)
    out_base, cyc_base, _ = simulate_kernel(base, {1: a_data, 2: b_data})

    unrolled = compile_kernel(uops, unroll_loop=4)
    out_unrolled, cyc_unrolled, _ = simulate_kernel(unrolled, {1: a_data, 2: b_data})

    np.testing.assert_array_equal(out_base, a_data + b_data)
    np.testing.assert_array_equal(out_unrolled, a_data + b_data)

    assert unrolled.compile_report["unroll_loop_applied"] == 4
    assert unrolled.compile_report["unroll_fallback_reason"] is None
    assert cyc_unrolled < cyc_base, f"Expected fewer cycles with unroll, got base={cyc_base}, unrolled={cyc_unrolled}"
    assert cyc_unrolled <= (cyc_base // 2), f"Expected significant speedup, got base={cyc_base}, unrolled={cyc_unrolled}"
