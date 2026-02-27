"""TDD test for element-wise multiplication: c = a * b."""

import os

os.environ["NOOPT"] = "1"
os.environ["DEBUG"] = "0"

import numpy as np
from tinygrad import Tensor, dtypes

from compiler.backend import _get_uops, analyze_buffers, compile_kernel, simulate_kernel
from compiler import HDLRenderer


def test_multiply_compile_and_simulate():
    """Compile and simulate: c = a * b where a, b, c are int32 vectors of length 4."""
    renderer = HDLRenderer()

    # Create tensors: c = a * b
    a = Tensor.empty(4, dtype=dtypes.int32)
    b = Tensor.empty(4, dtype=dtypes.int32)
    c = a * b

    sched = c.schedule()
    assert len(sched) == 1, "Should produce exactly one kernel"

    # Get UOps
    uops = _get_uops(sched[0].ast, renderer)

    # Analyze buffers
    bufs = analyze_buffers(uops)
    assert len(bufs) == 3, "Should have 3 buffers: output, a, b"

    # Compile to hardware
    kernel = compile_kernel(uops)
    assert kernel is not None

    # Test data
    a_data = np.array([1, 2, 3, 4], dtype=np.int32)
    b_data = np.array([5, 6, 7, 8], dtype=np.int32)
    expected = a_data * b_data  # [5, 12, 21, 32]

    # Simulate
    output, cycles, wall = simulate_kernel(kernel, {1: a_data, 2: b_data})

    # Verify
    assert output.shape == (4,), f"Output shape should be (4,), got {output.shape}"
    np.testing.assert_array_equal(
        output, expected, f"Expected {expected}, got {output}"
    )

    print(f"✓ Multiplication compiled and simulated successfully in {cycles} cycles")


def test_multiply_with_zeros():
    """Test multiplication with zero values."""
    renderer = HDLRenderer()

    a = Tensor.empty(4, dtype=dtypes.int32)
    b = Tensor.empty(4, dtype=dtypes.int32)
    c = a * b

    sched = c.schedule()
    uops = _get_uops(sched[0].ast, renderer)
    kernel = compile_kernel(uops)

    a_data = np.array([0, 1, 0, 5], dtype=np.int32)
    b_data = np.array([10, 20, 30, 40], dtype=np.int32)
    expected = np.array([0, 20, 0, 200], dtype=np.int32)

    output, _, _ = simulate_kernel(kernel, {1: a_data, 2: b_data})

    np.testing.assert_array_equal(output, expected)
    print(f"✓ Multiplication with zeros works correctly")


if __name__ == "__main__":
    test_multiply_compile_and_simulate()
    test_multiply_with_zeros()
