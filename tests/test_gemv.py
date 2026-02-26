"""Simulation testbench for INT8 GEMV unit."""

import numpy as np
import pytest
from amaranth.sim import Simulator

from hdl.gemv import GEMVUnit
from hdl.relu import ReLU


def run_gemv_sim(m_dim, k_dim, weights, vector):
    """Run GEMV simulation, return (results dict {idx: value}, cycle_count)."""
    dut = GEMVUnit(m_dim, k_dim)
    sim = Simulator(dut)
    sim.add_clock(1e-8)  # 100 MHz

    results = {}
    cycle_count = 0

    async def testbench(ctx):
        nonlocal cycle_count

        # Load vector
        for j in range(k_dim):
            ctx.set(dut.vec_wen, 1)
            ctx.set(dut.vec_waddr, j)
            ctx.set(dut.vec_wdata, int(vector[j]))
            await ctx.tick()
        ctx.set(dut.vec_wen, 0)

        # Load weights (row-major)
        for i in range(m_dim):
            for j in range(k_dim):
                ctx.set(dut.w_wen, 1)
                ctx.set(dut.w_waddr, i * k_dim + j)
                ctx.set(dut.w_wdata, int(weights[i, j]))
                await ctx.tick()
        ctx.set(dut.w_wen, 0)
        await ctx.tick()

        # Start computation
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)

        # Wait for results
        max_cycles = m_dim * k_dim + m_dim + 10
        for _ in range(max_cycles):
            await ctx.tick()
            cycle_count += 1

            if ctx.get(dut.result_valid):
                idx = ctx.get(dut.result_idx)
                data = ctx.get(dut.result_data)
                results[idx] = data

            if ctx.get(dut.done):
                break

    sim.add_testbench(testbench)
    with sim.write_vcd("gemv_test.vcd"):
        sim.run()

    return results, cycle_count


class TestGEMVSmall:
    """Small GEMV tests with known values."""

    def test_2x2(self):
        W = np.array([[1, 2], [3, 4]], dtype=np.int8)
        x = np.array([5, 6], dtype=np.int8)
        expected = W.astype(np.int32) @ x.astype(np.int32)  # [17, 39]

        results, _ = run_gemv_sim(2, 2, W, x)
        for i in range(2):
            assert results[i] == expected[i], f"y[{i}]: got {results[i]}, expected {expected[i]}"

    def test_4x3(self):
        W = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ], dtype=np.int8)
        x = np.array([1, 2, 3], dtype=np.int8)
        expected = W.astype(np.int32) @ x.astype(np.int32)  # [14, 32, 50, 68]

        results, _ = run_gemv_sim(4, 3, W, x)
        for i in range(4):
            assert results[i] == expected[i], f"y[{i}]: got {results[i]}, expected {expected[i]}"

    def test_negative_values(self):
        W = np.array([[-1, 2], [3, -4]], dtype=np.int8)
        x = np.array([-5, 6], dtype=np.int8)
        expected = W.astype(np.int32) @ x.astype(np.int32)  # [17, -39]

        results, _ = run_gemv_sim(2, 2, W, x)
        for i in range(2):
            assert results[i] == expected[i], f"y[{i}]: got {results[i]}, expected {expected[i]}"

    def test_identity(self):
        W = np.eye(3, dtype=np.int8)
        x = np.array([10, 20, 30], dtype=np.int8)
        expected = x.astype(np.int32)

        results, _ = run_gemv_sim(3, 3, W, x)
        for i in range(3):
            assert results[i] == expected[i], f"y[{i}]: got {results[i]}, expected {expected[i]}"

    def test_single_element(self):
        W = np.array([[7]], dtype=np.int8)
        x = np.array([3], dtype=np.int8)

        results, _ = run_gemv_sim(1, 1, W, x)
        assert results[0] == 21

    def test_random_8x16(self):
        rng = np.random.default_rng(42)
        W = rng.integers(-128, 127, size=(8, 16), dtype=np.int8)
        x = rng.integers(-128, 127, size=16, dtype=np.int8)
        expected = W.astype(np.int32) @ x.astype(np.int32)

        results, _ = run_gemv_sim(8, 16, W, x)
        for i in range(8):
            assert results[i] == expected[i], f"y[{i}]: got {results[i]}, expected {expected[i]}"


class TestGEMVTiming:
    """Timing and cycle count tests."""

    def test_cycle_count_4x3(self):
        W = np.ones((4, 3), dtype=np.int8)
        x = np.ones(3, dtype=np.int8)
        _, cycles = run_gemv_sim(4, 3, W, x)
        # K compute cycles + 1 emit cycle per row = M * (K + 1)
        expected_cycles = 4 * (3 + 1)
        assert cycles == expected_cycles, f"Got {cycles} cycles, expected {expected_cycles}"


class TestGEMVMNIST:
    """MNIST-sized GEMV to validate feasibility."""

    @pytest.mark.slow
    def test_kernel1_10x128(self):
        """Kernel 1: 10x128 GEMV — should complete in ~1,300 cycles."""
        rng = np.random.default_rng(123)
        M, K = 10, 128
        W = rng.integers(-128, 127, size=(M, K), dtype=np.int8)
        x = rng.integers(-128, 127, size=K, dtype=np.int8)
        expected = W.astype(np.int32) @ x.astype(np.int32)

        results, cycles = run_gemv_sim(M, K, W, x)
        for i in range(M):
            assert results[i] == expected[i], f"y[{i}]: got {results[i]}, expected {expected[i]}"

        print(f"\nKernel 1 ({M}x{K}): {cycles} compute cycles "
              f"({cycles * 10:.0f} ns at 100 MHz)")


class TestReLU:
    """ReLU block tests."""

    def test_relu_positive(self):
        dut = ReLU(32)
        sim = Simulator(dut)

        async def testbench(ctx):
            ctx.set(dut.inp, 42)
            assert ctx.get(dut.out) == 42

        sim.add_testbench(testbench)
        sim.run()

    def test_relu_negative(self):
        dut = ReLU(32)
        sim = Simulator(dut)

        async def testbench(ctx):
            # Set a negative value (-10 as two's complement in 32 bits)
            ctx.set(dut.inp, -10)
            assert ctx.get(dut.out) == 0

        sim.add_testbench(testbench)
        sim.run()

    def test_relu_zero(self):
        dut = ReLU(32)
        sim = Simulator(dut)

        async def testbench(ctx):
            ctx.set(dut.inp, 0)
            assert ctx.get(dut.out) == 0

        sim.add_testbench(testbench)
        sim.run()
