"""Tests for loop unrolling (compiler/transforms.py)."""

import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops

from compiler.backend import HDLRenderer, _get_uops, uops_to_kernel_ir, compile_kernel, simulate_kernel
from compiler.transforms import unroll_loop, unroll_reduce
from compiler.ir import IRConst, IRBufLoad, IRBufStore, IRRegStore, IROp, LoopIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_kir(build_fn):
    """Build a tinygrad computation and return (KernelIR, buf_infos, uops)."""
    result = build_fn()
    sched = result.schedule()
    renderer = HDLRenderer()
    uops = _get_uops(
        [s for s in sched if s.ast.op == Ops.SINK][-1].ast,
        renderer,
    )
    kir, buf_infos = uops_to_kernel_ir(uops)
    return kir, buf_infos, uops


def _make_uops(*arrays):
    """Get UOps for an elementwise add of two int32 arrays."""
    result = Tensor(arrays[0]) + Tensor(arrays[1])
    sched = result.schedule()
    return _get_uops(
        [s for s in sched if s.ast.op == Ops.SINK][-1].ast,
        HDLRenderer(),
    )


# ---------------------------------------------------------------------------
# IR transform tests
# ---------------------------------------------------------------------------

class TestUnrollIR:
    """Test that unroll_loop produces correct IR structure."""

    @pytest.fixture(scope="class")
    def vec3_kir(self):
        """Shared 3-element add KernelIR for tests that all use the same computation."""
        kir, _, _ = _get_kir(lambda: Tensor([1, 2, 3]) + Tensor([4, 5, 6]))
        return kir

    @pytest.fixture(scope="class")
    def vec6_kir(self):
        """Shared 6-element add KernelIR."""
        kir, _, _ = _get_kir(
            lambda: Tensor(np.arange(6, dtype=np.int32)) + Tensor(np.arange(10, 16, dtype=np.int32))
        )
        return kir

    def test_full_unroll_eliminates_loop(self, vec3_kir):
        """Full unroll (factor=bound) should eliminate the loop."""
        assert vec3_kir.loop_tree.body is not None  # has a loop
        result = unroll_loop(vec3_kir, 0, 3)
        assert result.loop_tree.body is None  # loop eliminated

    def test_full_unroll_produces_correct_store_count(self, vec3_kir):
        """Full unroll of N-element loop should produce N stores."""
        result = unroll_loop(vec3_kir, 0, 3)
        stores = [s for s in result.loop_tree.epilogue if isinstance(s, IRBufStore)]
        assert len(stores) == 3

    def test_full_unroll_addresses_are_constants(self, vec3_kir):
        """Full unroll should use constant addresses 0, 1, 2."""
        result = unroll_loop(vec3_kir, 0, 3)
        stores = [s for s in result.loop_tree.epilogue if isinstance(s, IRBufStore)]
        addrs = [s.addr for s in stores]
        assert all(isinstance(a, IRConst) for a in addrs)
        assert [a.value for a in addrs] == [0, 1, 2]

    def test_partial_unroll_reduces_bound(self, vec6_kir):
        """Partial unroll should reduce loop bound by factor."""
        assert vec6_kir.loop_tree.body.bound == 6
        result = unroll_loop(vec6_kir, 0, 2)
        assert result.loop_tree.body is not None
        assert result.loop_tree.body.bound == 3

    def test_partial_unroll_doubles_stores(self, vec6_kir):
        """Partial unroll by 2 should double the stores per iteration."""
        orig_stores = len([s for s in vec6_kir.loop_tree.body.prologue if isinstance(s, IRBufStore)])
        result = unroll_loop(vec6_kir, 0, 2)
        new_stores = len([s for s in result.loop_tree.body.prologue if isinstance(s, IRBufStore)])
        assert new_stores == orig_stores * 2

    def test_factor_must_divide_bound(self, vec3_kir):
        """Factor that doesn't divide bound should raise ValueError."""
        with pytest.raises(ValueError, match="does not divide"):
            unroll_loop(vec3_kir, 0, 2)

    def test_factor_1_is_noop(self, vec3_kir):
        """factor=1 should return the same KernelIR."""
        result = unroll_loop(vec3_kir, 0, 1)
        assert result is vec3_kir

    def test_invalid_depth_raises(self, vec3_kir):
        """Non-existent depth should raise ValueError."""
        with pytest.raises(ValueError, match="No loop at depth"):
            unroll_loop(vec3_kir, 5, 2)


# ---------------------------------------------------------------------------
# Simulation correctness tests
# ---------------------------------------------------------------------------

class TestUnrollSimulation:
    """Test that unrolled kernels produce correct simulation results."""

    def test_full_unroll_vec3_add(self):
        """Full unroll of 3-element vector add."""
        a = np.array([10, 20, 30], dtype=np.int32)
        b = np.array([1, 2, 3], dtype=np.int32)
        uops = _make_uops(a, b)
        ref = compile_kernel(uops, unroll_factor=1)
        out_ref, _, _ = simulate_kernel(ref, {1: a, 2: b})

        unrolled = compile_kernel(uops, unroll_factor=3)
        out_unrolled, _, _ = simulate_kernel(unrolled, {1: a, 2: b})

        np.testing.assert_array_equal(out_ref, out_unrolled)

    def test_partial_unroll_vec6_add(self):
        """Partial unroll (factor=2) of 6-element vector add."""
        a = np.arange(6, dtype=np.int32)
        b = np.arange(10, 16, dtype=np.int32)
        uops = _make_uops(a, b)
        ref = compile_kernel(uops, unroll_factor=1)
        out_ref, _, _ = simulate_kernel(ref, {1: a, 2: b})

        for factor in [2, 3, 6]:
            unrolled = compile_kernel(uops, unroll_factor=factor)
            out_u, _, _ = simulate_kernel(unrolled, {1: a, 2: b})
            np.testing.assert_array_equal(out_ref, out_u, err_msg=f"factor={factor}")

    def test_unroll_larger_elementwise(self):
        """Unroll a 12-element elementwise op (factors 2, 3, 4, 6, 12)."""
        a = np.arange(12, dtype=np.int32)
        b = np.ones(12, dtype=np.int32) * 5
        uops = _make_uops(a, b)
        kir, _ = uops_to_kernel_ir(uops)
        bound = kir.loop_tree.body.bound
        ref = compile_kernel(uops, unroll_factor=1)
        out_ref, _, _ = simulate_kernel(ref, {1: a, 2: b})

        for factor in [2, 3, 4, 6]:
            if bound % factor != 0:
                continue
            unrolled = compile_kernel(uops, unroll_factor=factor)
            out_u, _, _ = simulate_kernel(unrolled, {1: a, 2: b})
            np.testing.assert_array_equal(out_ref, out_u, err_msg=f"factor={factor}")

    def test_unroll_negative_values(self):
        """Ensure unrolling handles negative int32 values correctly."""
        a = np.array([-100, 50, -25, 75, 10, -60], dtype=np.int32)
        b = np.array([30, -20, 10, -40, 5, 15], dtype=np.int32)
        uops = _make_uops(a, b)
        kir, _ = uops_to_kernel_ir(uops)
        if kir.loop_tree.body is None:
            pytest.skip("tinygrad produced a scalar kernel (no loop to unroll)")
        ref = compile_kernel(uops, unroll_factor=1)
        out_ref, _, _ = simulate_kernel(ref, {1: a, 2: b})

        for factor in [2, 3, 6]:
            if kir.loop_tree.body.bound % factor != 0:
                continue
            unrolled = compile_kernel(uops, unroll_factor=factor)
            out_u, _, _ = simulate_kernel(unrolled, {1: a, 2: b})
            np.testing.assert_array_equal(out_ref, out_u, err_msg=f"factor={factor}")

    def test_compile_kernel_default_no_unroll(self):
        """compile_kernel with default unroll_factor=1 should be unchanged."""
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4, 5, 6], dtype=np.int32)
        uops = _make_uops(a, b)
        k = compile_kernel(uops)
        out, cycles, _ = simulate_kernel(k, {1: a, 2: b})
        np.testing.assert_array_equal(out, [5, 7, 9])
        assert cycles == 3  # 3-element loop, 1 cycle each


# ---------------------------------------------------------------------------
# REDUCE-axis unrolling — helpers
# ---------------------------------------------------------------------------

def _make_gemv_uops(M, K):
    """Get UOps for an int8 GEMV: x(K,) @ w(M,K).T → out(M,)."""
    x = Tensor.empty(K, dtype=dtypes.int8)
    w = Tensor.empty(M, K, dtype=dtypes.int8)
    result = x @ w.T
    sched = result.schedule()
    return _get_uops(
        [s for s in sched if s.ast.op == Ops.SINK][-1].ast,
        HDLRenderer(),
    )


# ---------------------------------------------------------------------------
# REDUCE-axis unrolling — IR tests
# ---------------------------------------------------------------------------

class TestReduceUnrollIR:
    """Test that unroll_reduce produces correct IR structure."""

    @pytest.fixture(scope="class")
    def gemv_kir(self):
        """4x3 GEMV KernelIR: outer LOOP bound=4, inner REDUCE bound=3."""
        kir, _, _ = _get_kir(
            lambda: Tensor.empty(3, dtype=dtypes.int8) @ Tensor.empty(4, 3, dtype=dtypes.int8).T
        )
        return kir

    def test_reduce_unroll_reduces_bound(self, gemv_kir):
        """Full REDUCE unroll should reduce inner bound to 1."""
        result = unroll_reduce(gemv_kir, depth=1, factor=3)
        assert result.loop_tree.body.body.bound == 1

    def test_reduce_unroll_preserves_outer_loop(self, gemv_kir):
        """Outer LOOP bound should be unchanged."""
        result = unroll_reduce(gemv_kir, depth=1, factor=3)
        assert result.loop_tree.body.bound == 4

    def test_reduce_unroll_increases_bufloads(self, gemv_kir):
        """After unroll by 3, there should be 3x as many IRBufLoad nodes."""
        from compiler.ir import IRBufLoad

        def count_loads(stores):
            count = 0
            visited = set()
            def walk(val):
                nonlocal count
                if val is None or id(val) in visited:
                    return
                visited.add(id(val))
                if isinstance(val, IRBufLoad):
                    count += 1
                    walk(val.addr)
                elif isinstance(val, IROp):
                    for s in val.srcs:
                        walk(s)
            for s in stores:
                if hasattr(s, 'value'):
                    walk(s.value)
            return count

        orig_loads = count_loads(gemv_kir.loop_tree.body.body.prologue)
        result = unroll_reduce(gemv_kir, depth=1, factor=3)
        new_loads = count_loads(result.loop_tree.body.body.prologue)
        assert new_loads == orig_loads * 3

    def test_reject_loop_axis(self, gemv_kir):
        """Should reject LOOP axis with clear error."""
        with pytest.raises(ValueError, match="REDUCE"):
            unroll_reduce(gemv_kir, depth=0, factor=2)

    def test_factor_must_divide_bound(self, gemv_kir):
        """Factor that doesn't divide REDUCE bound should raise."""
        with pytest.raises(ValueError, match="does not divide"):
            unroll_reduce(gemv_kir, depth=1, factor=2)  # 3 % 2 != 0

    def test_factor_1_noop(self, gemv_kir):
        """factor=1 should return the same KernelIR."""
        result = unroll_reduce(gemv_kir, depth=1, factor=1)
        assert result is gemv_kir


# ---------------------------------------------------------------------------
# REDUCE-axis unrolling — simulation tests
# ---------------------------------------------------------------------------

class TestReduceUnrollSimulation:
    """Test that REDUCE-unrolled kernels produce correct results."""

    def test_gemv_full_reduce_unroll(self):
        """GEMV 4x3, full unroll of REDUCE (factor=3)."""
        x = np.array([1, 2, 3], dtype=np.int8)
        w = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1]], dtype=np.int8)
        uops = _make_gemv_uops(4, 3)

        ref = compile_kernel(uops, reduce_unroll_factor=1)
        out_ref, _, _ = simulate_kernel(ref, {1: x, 2: w.flatten()})

        unrolled = compile_kernel(uops, reduce_unroll_factor=3)
        out_u, _, _ = simulate_kernel(unrolled, {1: x, 2: w.flatten()})

        np.testing.assert_array_equal(out_ref, out_u)

    def test_gemv_partial_reduce_unroll(self):
        """GEMV 2x12, partial unroll with multiple factors."""
        x = np.arange(12, dtype=np.int8)
        w = np.ones((2, 12), dtype=np.int8)
        uops = _make_gemv_uops(2, 12)

        ref = compile_kernel(uops, reduce_unroll_factor=1)
        out_ref, _, _ = simulate_kernel(ref, {1: x, 2: w.flatten()})

        for factor in [2, 3, 4, 6, 12]:
            unrolled = compile_kernel(uops, reduce_unroll_factor=factor)
            out_u, _, _ = simulate_kernel(unrolled, {1: x, 2: w.flatten()})
            np.testing.assert_array_equal(out_ref, out_u, err_msg=f"factor={factor}")

    def test_reduce_unroll_cycle_count(self):
        """Verify cycles match M*(K/N+2)+1-1 formula."""
        M, K = 2, 12
        uops = _make_gemv_uops(M, K)

        ref = compile_kernel(uops, reduce_unroll_factor=1)
        _, cyc_ref, _ = simulate_kernel(ref, {1: np.zeros(K, dtype=np.int8),
                                               2: np.zeros(M*K, dtype=np.int8)})
        assert cyc_ref == M * (K + 2)  # 2*(12+2) = 28

        for factor in [2, 3, 4, 6, 12]:
            unrolled = compile_kernel(uops, reduce_unroll_factor=factor)
            _, cyc_u, _ = simulate_kernel(unrolled, {1: np.zeros(K, dtype=np.int8),
                                                      2: np.zeros(M*K, dtype=np.int8)})
            expected = M * (K // factor + 2)
            assert cyc_u == expected, f"factor={factor}: got {cyc_u}, expected {expected}"
