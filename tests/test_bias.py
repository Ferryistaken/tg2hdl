"""TDD test for addition with bias: c = a + b + bias."""

import os

os.environ["NOOPT"] = "1"
os.environ["DEBUG"] = "0"

import numpy as np
from tinygrad import Tensor, dtypes

from compiler.backend import _get_uops, analyze_buffers, compile_kernel, simulate_kernel
from compiler import HDLRenderer


def test_add_with_bias():
    """Compile and simulate: c = a + b + bias where bias is a scalar constant."""
    renderer = HDLRenderer()

    # Create tensors: c = a + b + bias
    a = Tensor.empty(4, dtype=dtypes.int32)
    b = Tensor.empty(4, dtype=dtypes.int32)
    bias = Tensor(5, dtype=dtypes.int32)  # scalar bias
    c = a + b + bias

    sched = c.schedule()
    assert len(sched) == 1, "Should produce exactly one kernel"

    # Get UOps
    uops = _get_uops(sched[-1].ast, renderer)

    # Analyze buffers - only 3 buffers (output, a, b), bias is a CONST
    bufs = analyze_buffers(uops)
    assert len(bufs) == 3, "Should have 3 buffers: output, a, b"

    # Compile to hardware
    kernel = compile_kernel(uops)
    assert kernel is not None

    # Test data
    a_data = np.array([1, 2, 3, 4], dtype=np.int32)
    b_data = np.array([10, 20, 30, 40], dtype=np.int32)
    expected = a_data + b_data + 5  # [16, 27, 38, 49]

    # Simulate - only pass buffers 1 and 2 (bias is CONST, not a buffer)
    output, cycles, wall = simulate_kernel(kernel, {1: a_data, 2: b_data})

    # Verify
    assert output.shape == (4,), f"Output shape should be (4,), got {output.shape}"
    np.testing.assert_array_equal(
        output, expected, f"Expected {expected}, got {output}"
    )

    print(
        f"✓ Addition with bias compiled and simulated successfully in {cycles} cycles"
    )


def test_add_with_negative_bias():
    """Test addition with negative bias value."""
    renderer = HDLRenderer()

    a = Tensor.empty(4, dtype=dtypes.int32)
    b = Tensor.empty(4, dtype=dtypes.int32)
    bias = Tensor(-3, dtype=dtypes.int32)
    c = a + b + bias

    sched = c.schedule()
    uops = _get_uops(sched[-1].ast, renderer)
    kernel = compile_kernel(uops)

    a_data = np.array([10, 20, 30, 40], dtype=np.int32)
    b_data = np.array([5, 5, 5, 5], dtype=np.int32)
    expected = np.array([12, 22, 32, 42], dtype=np.int32)

    output, _, _ = simulate_kernel(kernel, {1: a_data, 2: b_data})

    np.testing.assert_array_equal(output, expected)
    print(f"✓ Addition with negative bias works correctly")


if __name__ == "__main__":
    test_add_with_bias()
    test_add_with_negative_bias()
