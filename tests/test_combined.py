"""TDD test for combined operations: c = relu(a + b + bias)."""

import os

os.environ["NOOPT"] = "1"
os.environ["DEBUG"] = "0"

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops

from compiler.backend import _get_uops, analyze_buffers, compile_kernel, simulate_kernel
from compiler import HDLRenderer


def test_relu_add_bias_compile_and_simulate():
    """Compile and simulate: c = relu(a + b + bias)."""
    renderer = HDLRenderer()

    # Create tensors: c = relu(a + b + bias)
    a = Tensor.empty(4, dtype=dtypes.int32)
    b = Tensor.empty(4, dtype=dtypes.int32)
    bias = Tensor(5, dtype=dtypes.int32)
    c = (a + b + bias).relu()

    sched = c.schedule()
    assert len(sched) == 1, "Should produce exactly one kernel"

    # Get UOps
    uops = _get_uops(sched[-1].ast, renderer)

    # Analyze buffers
    bufs = analyze_buffers(uops)
    assert len(bufs) == 3, "Should have 3 buffers: output, a, b (bias is CONST)"

    # Verify UOp structure
    op_types = [uop.op for uop in uops]
    assert Ops.ADD in op_types, "Should have ADD operation"
    assert Ops.CMPLT in op_types, "Should have CMPLT for ReLU comparison"
    assert Ops.WHERE in op_types, "Should have WHERE for ReLU ternary"

    # Compile to hardware
    kernel = compile_kernel(uops)
    assert kernel is not None

    # Verify kernel structure
    assert len(kernel.buf_infos) == 3, "Should have 3 buffer infos"

    # Test data
    a_data = np.array([-10, 0, 5, 10], dtype=np.int32)
    b_data = np.array([3, 5, 2, -5], dtype=np.int32)
    # a + b + 5 = [-2, 10, 12, 10]
    # relu = [0, 10, 12, 10]
    expected = np.array([0, 10, 12, 10], dtype=np.int32)

    # Simulate
    output, cycles, wall = simulate_kernel(kernel, {1: a_data, 2: b_data})

    # Verify
    assert output.shape == (4,), f"Output shape should be (4,), got {output.shape}"
    np.testing.assert_array_equal(
        output, expected, f"Expected {expected}, got {output}"
    )

    # Verify cycle count: N compute cycles
    assert cycles == 4, f"Should take 4 cycles, took {cycles}"

    print(f"✓ relu(a + b + bias) compiled and simulated in {cycles} cycles")


def test_relu_add_bias_compilation_structure():
    """Test that compilation produces correct structure."""
    renderer = HDLRenderer()

    a = Tensor.empty(8, dtype=dtypes.int32)
    b = Tensor.empty(8, dtype=dtypes.int32)
    bias = Tensor(-3, dtype=dtypes.int32)
    c = (a + b + bias).relu()

    sched = c.schedule()
    uops = _get_uops(sched[-1].ast, renderer)
    kernel = compile_kernel(uops)

    # Verify UOp counts
    op_counts = {}
    for uop in uops:
        op_counts[uop.op] = op_counts.get(uop.op, 0) + 1

    # Should have: 3 DEFINE_GLOBALs, 1 CONST for loop range, 1 CONST for bias
    assert op_counts.get(Ops.DEFINE_GLOBAL, 0) == 3
    assert op_counts.get(Ops.CONST, 0) >= 2  # loop range + bias value

    # Should have: 2 LOADs (for a and b), 2 ADDs (a+b, then +bias)
    assert op_counts.get(Ops.LOAD, 0) == 2
    assert op_counts.get(Ops.ADD, 0) >= 1

    # Should have: 1 CMPLT, 1 WHERE for ReLU
    assert op_counts.get(Ops.CMPLT, 0) == 1
    assert op_counts.get(Ops.WHERE, 0) == 1

    # Should have: 1 STORE
    assert op_counts.get(Ops.STORE, 0) == 1

    # Verify memory depths
    for idx, buf in enumerate(kernel.buf_infos):
        if buf["is_output"]:
            assert buf["depth"] == 8, f"Output memory should have depth 8, got {buf['depth']}"
        else:
            assert buf["depth"] == 8, (
                f"Input memory {idx} should have depth 8, got {buf['depth']}"
            )

    print(f"✓ Compilation structure is correct")


def test_relu_add_bias_edge_cases():
    """Test edge cases: all zeros, max values, etc."""
    renderer = HDLRenderer()

    a = Tensor.empty(4, dtype=dtypes.int32)
    b = Tensor.empty(4, dtype=dtypes.int32)
    bias = Tensor(0, dtype=dtypes.int32)
    c = (a + b + bias).relu()

    sched = c.schedule()
    uops = _get_uops(sched[-1].ast, renderer)
    kernel = compile_kernel(uops)

    # Test with zeros
    a_data = np.array([0, 0, 0, 0], dtype=np.int32)
    b_data = np.array([0, 0, 0, 0], dtype=np.int32)
    expected = np.array([0, 0, 0, 0], dtype=np.int32)

    output, _, _ = simulate_kernel(kernel, {1: a_data, 2: b_data})
    np.testing.assert_array_equal(output, expected)

    # Test with large values
    a_data = np.array([1000000, -1000000, 2000000, -2000000], dtype=np.int32)
    b_data = np.array([-500000, 500000, -1000000, 1000000], dtype=np.int32)
    expected = np.array([500000, 0, 1000000, 0], dtype=np.int32)

    output, _, _ = simulate_kernel(kernel, {1: a_data, 2: b_data})
    np.testing.assert_array_equal(output, expected)

    print(f"✓ Edge cases handled correctly")


def test_relu_add_bias_cycle_count():
    """Test that cycle count matches expected (N cycles for element-wise)."""
    renderer = HDLRenderer()

    for n in [1, 4, 8, 16, 32]:
        a = Tensor.empty(n, dtype=dtypes.int32)
        b = Tensor.empty(n, dtype=dtypes.int32)
        bias = Tensor(1, dtype=dtypes.int32)
        c = (a + b + bias).relu()

        sched = c.schedule()
        uops = _get_uops(sched[-1].ast, renderer)
        kernel = compile_kernel(uops)

        a_data = np.arange(n, dtype=np.int32)
        b_data = np.arange(n, dtype=np.int32)

        output, cycles, wall = simulate_kernel(kernel, {1: a_data, 2: b_data})

        # Should take exactly N cycles for element-wise
        assert cycles == n, f"For n={n}, expected {n} cycles, got {cycles}"

    print(f"✓ Cycle counts are optimal (N cycles for N elements)")


if __name__ == "__main__":
    test_relu_add_bias_compile_and_simulate()
    test_relu_add_bias_compilation_structure()
    test_relu_add_bias_edge_cases()
    test_relu_add_bias_cycle_count()
