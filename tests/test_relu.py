"""TDD test for ReLU activation: c = relu(x)."""

import os

os.environ["NOOPT"] = "1"
os.environ["DEBUG"] = "0"

import numpy as np
from tinygrad import Tensor, dtypes

from compiler.backend import _get_uops, analyze_buffers, compile_kernel, simulate_kernel
from compiler import HDLRenderer


def test_relu_compile_and_simulate():
    """Compile and simulate: c = relu(x) where x is int32 vector."""
    renderer = HDLRenderer()

    # Create tensors: c = relu(x)
    x = Tensor.empty(4, dtype=dtypes.int32)
    c = x.relu()

    sched = c.schedule()
    assert len(sched) == 1, "Should produce exactly one kernel"

    # Get UOps
    uops = _get_uops(sched[-1].ast, renderer)

    # Analyze buffers
    bufs = analyze_buffers(uops)
    assert len(bufs) == 2, "Should have 2 buffers: output, x"

    # Compile to hardware
    kernel = compile_kernel(uops)
    assert kernel is not None

    # Test data
    x_data = np.array([-5, 0, 3, 10], dtype=np.int32)
    expected = np.array([0, 0, 3, 10], dtype=np.int32)

    # Simulate
    output, cycles, wall = simulate_kernel(kernel, {1: x_data})

    # Verify
    assert output.shape == (4,), f"Output shape should be (4,), got {output.shape}"
    np.testing.assert_array_equal(
        output, expected, f"Expected {expected}, got {output}"
    )

    print(f"✓ ReLU compiled and simulated successfully in {cycles} cycles")


def test_relu_all_positive():
    """Test ReLU with all positive values."""
    renderer = HDLRenderer()

    x = Tensor.empty(4, dtype=dtypes.int32)
    c = x.relu()

    sched = c.schedule()
    uops = _get_uops(sched[-1].ast, renderer)
    kernel = compile_kernel(uops)

    x_data = np.array([1, 2, 3, 4], dtype=np.int32)
    expected = np.array([1, 2, 3, 4], dtype=np.int32)

    output, _, _ = simulate_kernel(kernel, {1: x_data})

    np.testing.assert_array_equal(output, expected)
    print(f"✓ ReLU with all positive values works correctly")


def test_relu_all_negative():
    """Test ReLU with all negative values."""
    renderer = HDLRenderer()

    x = Tensor.empty(4, dtype=dtypes.int32)
    c = x.relu()

    sched = c.schedule()
    uops = _get_uops(sched[-1].ast, renderer)
    kernel = compile_kernel(uops)

    x_data = np.array([-10, -5, -1, -100], dtype=np.int32)
    expected = np.array([0, 0, 0, 0], dtype=np.int32)

    output, _, _ = simulate_kernel(kernel, {1: x_data})

    np.testing.assert_array_equal(output, expected)
    print(f"✓ ReLU with all negative values works correctly")


if __name__ == "__main__":
    test_relu_compile_and_simulate()
    test_relu_all_positive()
    test_relu_all_negative()
