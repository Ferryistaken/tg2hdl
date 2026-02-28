"""Tests for the tinygrad UOps → HDL compiler."""

import os

os.environ["NOOPT"] = "1"
os.environ["DEBUG"] = "0"

import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, KernelInfo

from tinygrad.uop.ops import AxisType

from compiler import HDLRenderer, compile_kernel, simulate_kernel
from compiler.backend import _get_uops, analyze_buffers
from compiler.hdl_module import LoopLevel


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------

class TestHDLRenderer:
    def test_renderer_produces_clean_uops(self):
        """HDLRenderer should produce UOps without GPU-specific ops."""
        renderer = HDLRenderer()
        x = Tensor.empty(1, 3, dtype=dtypes.int8)
        w = Tensor.empty(3, 4, dtype=dtypes.int8)
        out = (x @ w).cast(dtypes.int32)
        sched = out.schedule()

        for si in sched:
            if si.ast.op != Ops.SINK:
                continue
            uops = _get_uops(si.ast, renderer)
            gpu_ops = {Ops.SPECIAL, Ops.DEFINE_LOCAL}
            for u in uops:
                assert u.op not in gpu_ops, f"GPU op {u.op} in sequential UOps"

    def test_renderer_attributes(self):
        """HDLRenderer should disable GPU features."""
        r = HDLRenderer()
        assert r.has_local is False
        assert r.has_shared is False
        assert r.supports_float4 is False
        assert r.global_max is None
        assert r.local_max is None


# ---------------------------------------------------------------------------
# Buffer analysis tests
# ---------------------------------------------------------------------------

class TestBufferAnalysis:
    def test_matmul_buffers(self):
        """4x3 matmul should have 3 buffers: output, input, weights."""
        renderer = HDLRenderer()
        x = Tensor.empty(1, 3, dtype=dtypes.int8)
        w = Tensor.empty(3, 4, dtype=dtypes.int8)
        out = (x @ w).cast(dtypes.int32)
        sched = out.schedule()

        ast = sched[0].ast
        uops = _get_uops(ast, renderer)
        bufs = analyze_buffers(uops)

        assert len(bufs) == 3
        # Buffer 0 is output (int32, depth 4)
        assert bufs[0].idx == 0
        assert bufs[0].is_output is True
        assert bufs[0].elem_width == 32
        assert bufs[0].depth == 4
        # Buffer 1 is input (int8, depth 3)
        assert bufs[1].idx == 1
        assert bufs[1].elem_width == 8
        assert bufs[1].depth == 3
        # Buffer 2 is weights (int8, depth 12)
        assert bufs[2].idx == 2
        assert bufs[2].elem_width == 8
        assert bufs[2].depth == 12


# ---------------------------------------------------------------------------
# Single kernel compilation + simulation tests
# ---------------------------------------------------------------------------

class TestSingleKernel:
    def test_compile_small_matmul(self):
        """Compiling a 4×3 matmul should produce a CompiledKernel."""
        renderer = HDLRenderer()
        x = Tensor.empty(1, 3, dtype=dtypes.int8)
        w = Tensor.empty(3, 4, dtype=dtypes.int8)
        out = (x @ w).cast(dtypes.int32)
        sched = out.schedule()

        uops = _get_uops(sched[0].ast, renderer)
        kernel = compile_kernel(uops)

        assert kernel is not None
        assert hasattr(kernel, "start")
        assert hasattr(kernel, "done")
        assert 0 in kernel.buf_write_ports
        assert 0 in kernel.buf_read_ports

    def test_simulate_identity_matmul(self):
        """Simulate matmul with identity-like weights: y = x @ I."""
        renderer = HDLRenderer()

        # 3×3 identity matmul
        x = Tensor.empty(1, 3, dtype=dtypes.int8)
        w = Tensor.empty(3, 3, dtype=dtypes.int8)
        out = (x @ w).cast(dtypes.int32)
        sched = out.schedule()

        uops = _get_uops(sched[0].ast, renderer)
        kernel = compile_kernel(uops)

        # Input vector: [1, 2, 3]
        x_data = np.array([1, 2, 3], dtype=np.int8)
        # Identity weights (3x3, row-major from tinygrad's perspective)
        # tinygrad accesses as w[col*rows + row], which is column-major
        # For identity: w[j*3 + i] = 1 if i==j else 0
        w_data = np.eye(3, dtype=np.int8).T.flatten()  # transpose for column-major

        output, cycles, wall = simulate_kernel(kernel, {1: x_data, 2: w_data})

        # Output should be [1, 2, 3] (modulo int8 truncation from CASTs)
        assert output.shape == (3,)
        np.testing.assert_array_equal(output, x_data.astype(np.int32))

    def test_simulate_small_matmul_4x3(self):
        """Simulate a 4×3 matmul and compare to numpy."""
        renderer = HDLRenderer()

        x = Tensor.empty(1, 3, dtype=dtypes.int8)
        w = Tensor.empty(3, 4, dtype=dtypes.int8)
        out = (x @ w).cast(dtypes.int32)
        sched = out.schedule()

        uops = _get_uops(sched[0].ast, renderer)
        kernel = compile_kernel(uops)

        # Test data
        x_np = np.array([10, 20, 30], dtype=np.int8)
        # Weight matrix shape (3, 4) in tinygrad, stored column-major for matmul
        w_np = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ], dtype=np.int8)

        # tinygrad stores w as w[j*4 + i] where j is reduction dim, i is output dim
        w_flat = w_np.flatten()  # already row-major: w[j][i]

        # Expected: x @ w, but with int8 truncation in the accumulator path
        # The UOps do: acc (int32) += cast_int32(cast_int8(x[j] * w[j*4+i]))
        # So each multiply is truncated to int8 before accumulation
        expected = np.zeros(4, dtype=np.int32)
        for i in range(4):
            acc = np.int32(0)
            for j in range(3):
                product = np.int8(np.int8(x_np[j]) * np.int8(w_np[j, i]))
                acc += np.int32(product)
            # Output goes through cast int→char→int (truncate to int8, sign extend)
            acc = np.int32(np.int8(acc))
            expected[i] = acc

        output, cycles, wall = simulate_kernel(kernel, {1: x_np, 2: w_flat})
        np.testing.assert_array_equal(output, expected)

    def test_simulate_matmul_with_bias_relu(self):
        """Simulate matmul + bias + ReLU (like MNIST kernel 0)."""
        renderer = HDLRenderer()

        x = Tensor.empty(1, 3, dtype=dtypes.int8)
        w = Tensor.empty(3, 4, dtype=dtypes.int8)
        b = Tensor.empty(1, 4, dtype=dtypes.int32)
        out = ((x @ w).cast(dtypes.int32) + b).relu()
        sched = out.schedule()

        uops = _get_uops(sched[0].ast, renderer)
        kernel = compile_kernel(uops)

        # Test data
        x_np = np.array([1, 2, 3], dtype=np.int8)
        w_np = np.array([
            [1, -1, 2, -2],
            [3, -3, 4, -4],
            [5, -5, 6, -6],
        ], dtype=np.int8)
        b_np = np.array([10, 20, -100, 5], dtype=np.int32)

        w_flat = w_np.flatten()

        # Expected computation matching UOp semantics:
        # For each output i:
        #   acc = sum_j(int32(int8(x[j] * w[j,i])))
        #   result = relu(int32(int8(int32(int8(acc)) + bias[i])))
        expected = np.zeros(4, dtype=np.int32)
        for i in range(4):
            acc = np.int32(0)
            for j in range(3):
                product = np.int8(np.int8(x_np[j]) * np.int8(w_np[j, i]))
                acc += np.int32(product)
            trunc = np.int32(np.int8(acc))
            with_bias = trunc + b_np[i]
            # ReLU: WHERE(0 < with_bias, int32(int8(with_bias)), 0)
            if with_bias > 0:
                expected[i] = np.int32(np.int8(with_bias))
            else:
                expected[i] = 0

        output, cycles, wall = simulate_kernel(
            kernel,
            {1: x_np, 2: w_flat, 3: b_np},
        )
        np.testing.assert_array_equal(output, expected)

    def test_cycle_count(self):
        """Verify cycle count is approximately M × (K + 2)."""
        renderer = HDLRenderer()

        M, K = 4, 3
        x = Tensor.empty(1, K, dtype=dtypes.int8)
        w = Tensor.empty(K, M, dtype=dtypes.int8)
        out = (x @ w).cast(dtypes.int32)
        sched = out.schedule()

        uops = _get_uops(sched[0].ast, renderer)
        kernel = compile_kernel(uops)

        x_np = np.ones(K, dtype=np.int8)
        w_np = np.ones(M * K, dtype=np.int8)

        _, cycles, _ = simulate_kernel(kernel, {1: x_np, 2: w_np})

        # FSM: M iterations of (1 INIT_ROW + K COMPUTE + 1 POST) + 1 DONE
        expected_cycles = M * (K + 2) + 1
        assert abs(cycles - expected_cycles) <= 2, (
            f"Expected ~{expected_cycles} cycles, got {cycles}"
        )


# ---------------------------------------------------------------------------
# Loop structure parser tests
# ---------------------------------------------------------------------------

class TestParseLoopStructure:
    def _parse(self, uops):
        """Helper: call _parse_loop_structure via a kernel instance."""
        kernel = compile_kernel(uops)
        return kernel._parse_loop_structure(uops)

    def test_scalar_returns_root_no_body(self):
        """Kernel with no loops (n=1 elementwise) returns root with body=None."""
        renderer = HDLRenderer()
        a = Tensor.empty(1, dtype=dtypes.int32)
        b = Tensor.empty(1, dtype=dtypes.int32)
        c = (a + b).relu()
        uops = _get_uops(c.schedule()[-1].ast, renderer)
        result = self._parse(uops)
        assert result is not None          # always returns root
        assert result.axis_type is None    # root level
        assert result.body is None         # no loops

    def test_elementwise_one_loop(self):
        """Simple elementwise op produces root with one LOOP level, no body."""
        renderer = HDLRenderer()
        a = Tensor.empty(4, dtype=dtypes.int32)
        b = Tensor.empty(4, dtype=dtypes.int32)
        c = a + b
        uops = _get_uops(c.schedule()[-1].ast, renderer)
        result = self._parse(uops)

        assert result is not None
        assert result.axis_type is None    # root
        assert result.body is not None     # has a loop
        assert result.body.axis_type == AxisType.LOOP
        assert result.body.bound == 4
        assert result.body.body is None
        assert len(result.body.prologue) > 0   # loads, add, store
        assert len(result.body.epilogue) == 0

    def test_gemv_two_levels(self):
        """Matmul produces root → outer LOOP → inner REDUCE."""
        renderer = HDLRenderer()
        x = Tensor.empty(3, dtype=dtypes.int8)
        w = Tensor.empty(4, 3, dtype=dtypes.int8)
        out = x @ w.T
        uops = _get_uops(out.schedule()[-1].ast, renderer)
        result = self._parse(uops)

        assert result is not None
        assert result.axis_type is None    # root
        outer = result.body
        assert outer is not None
        assert outer.axis_type == AxisType.LOOP
        assert outer.bound == 4
        assert outer.body is not None
        assert outer.body.axis_type == AxisType.REDUCE
        assert outer.body.bound == 3
        assert outer.body.body is None

    def test_gemv_prologue_epilogue_split(self):
        """Outer level prologue has acc reset, epilogue has output store."""
        renderer = HDLRenderer()
        x = Tensor.empty(3, dtype=dtypes.int8)
        w = Tensor.empty(4, 3, dtype=dtypes.int8)
        out = x @ w.T
        uops = _get_uops(out.schedule()[-1].ast, renderer)
        result = self._parse(uops)
        outer = result.body

        # prologue: ops before inner RANGE (acc reset)
        assert len(outer.prologue) > 0
        prologue_ops = {u.op for u in outer.prologue}
        assert Ops.STORE in prologue_ops   # accumulator reset

        # epilogue: ops after inner END (read acc, store to output)
        assert len(outer.epilogue) > 0
        epilogue_ops = {u.op for u in outer.epilogue}
        assert Ops.STORE in epilogue_ops   # output write
        assert Ops.LOAD in epilogue_ops    # read final acc


# ---------------------------------------------------------------------------
# Multi-kernel chaining tests
# ---------------------------------------------------------------------------

class TestMultiKernel:
    def test_two_layer_mlp_simulation(self):
        """Simulate a 2-layer MLP: (1,4)→(1,3)→(1,2) with real data.

        Compiles two kernels from one tinygrad graph, chains them by
        feeding kernel 0's output as kernel 1's input, and compares
        the final result to a numpy reference.
        """
        renderer = HDLRenderer()

        # Build tinygrad graph: 2-layer MLP with bias + relu
        x_sym = Tensor.empty(1, 4, dtype=dtypes.int8)
        w1_sym = Tensor.empty(4, 3, dtype=dtypes.int8)
        b1_sym = Tensor.empty(1, 3, dtype=dtypes.int32)
        w2_sym = Tensor.empty(3, 2, dtype=dtypes.int8)
        b2_sym = Tensor.empty(1, 2, dtype=dtypes.int32)

        h_sym = ((x_sym @ w1_sym).cast(dtypes.int32) + b1_sym).relu()
        h_i8 = h_sym.cast(dtypes.int8)
        logits_sym = (h_i8 @ w2_sym).cast(dtypes.int32) + b2_sym
        sched = logits_sym.schedule()

        # Should produce 2 compute kernels
        compute_items = [si for si in sched if si.ast.op == Ops.SINK]
        assert len(compute_items) == 2

        # Compile both kernels
        uops0 = _get_uops(compute_items[0].ast, renderer)
        uops1 = _get_uops(compute_items[1].ast, renderer)
        kernel0 = compile_kernel(uops0)
        kernel1 = compile_kernel(uops1)

        # Test data
        x_np = np.array([1, 2, 3, 4], dtype=np.int8)
        w1_np = np.array([
            [1,  0, -1],
            [0,  1,  0],
            [1,  1,  1],
            [0,  0,  1],
        ], dtype=np.int8)  # (4, 3)
        b1_np = np.array([5, -10, 0], dtype=np.int32)

        w2_np = np.array([
            [ 1, -1],
            [ 2,  0],
            [ 0,  1],
        ], dtype=np.int8)  # (3, 2)
        b2_np = np.array([3, -3], dtype=np.int32)

        # --- Numpy reference (matching UOp INT8 truncation semantics) ---
        # Layer 1: GEMV + bias + relu with int8 truncation
        h_ref = np.zeros(3, dtype=np.int32)
        for i in range(3):
            acc = np.int32(0)
            for j in range(4):
                product = np.int8(np.int8(x_np[j]) * np.int8(w1_np[j, i]))
                acc += np.int32(product)
            trunc = np.int32(np.int8(acc))
            with_bias = trunc + b1_np[i]
            if with_bias > 0:
                h_ref[i] = np.int32(np.int8(with_bias))
            else:
                h_ref[i] = 0

        # Layer 2: kernel reads int32, casts to int8 internally, then GEMV + bias
        logits_ref = np.zeros(2, dtype=np.int32)
        for i in range(2):
            acc = np.int32(0)
            for j in range(3):
                h_val = np.int8(h_ref[j])  # cast fused in kernel 1
                product = np.int8(h_val * np.int8(w2_np[j, i]))
                acc += np.int32(product)
            trunc = np.int32(np.int8(acc))
            logits_ref[i] = trunc + b2_np[i]

        # --- Simulate kernel 0 ---
        out0, cycles0, _ = simulate_kernel(
            kernel0, {1: x_np, 2: w1_np.flatten(), 3: b1_np}
        )
        np.testing.assert_array_equal(out0, h_ref,
            err_msg="Kernel 0 (layer 1) output mismatch")

        # --- Chain: feed kernel 0 output into kernel 1 ---
        out1, cycles1, _ = simulate_kernel(
            kernel1, {1: out0, 2: w2_np.flatten(), 3: b2_np}
        )
        np.testing.assert_array_equal(out1, logits_ref,
            err_msg="Kernel 1 (layer 2) output mismatch")

    def test_two_layer_mlp_prediction(self):
        """2-layer MLP should produce a valid argmax prediction."""
        renderer = HDLRenderer()

        # Slightly larger: (1,8) → (1,4) → (1,3)
        x_sym = Tensor.empty(1, 8, dtype=dtypes.int8)
        w1_sym = Tensor.empty(8, 4, dtype=dtypes.int8)
        b1_sym = Tensor.empty(1, 4, dtype=dtypes.int32)
        w2_sym = Tensor.empty(4, 3, dtype=dtypes.int8)
        b2_sym = Tensor.empty(1, 3, dtype=dtypes.int32)

        h_sym = ((x_sym @ w1_sym).cast(dtypes.int32) + b1_sym).relu()
        h_i8 = h_sym.cast(dtypes.int8)
        logits_sym = (h_i8 @ w2_sym).cast(dtypes.int32) + b2_sym
        sched = logits_sym.schedule()

        compute_items = [si for si in sched if si.ast.op == Ops.SINK]
        assert len(compute_items) == 2

        uops0 = _get_uops(compute_items[0].ast, renderer)
        uops1 = _get_uops(compute_items[1].ast, renderer)
        kernel0 = compile_kernel(uops0)
        kernel1 = compile_kernel(uops1)

        # Random but deterministic test data
        rng = np.random.RandomState(42)
        x_np = rng.randint(-10, 10, size=8).astype(np.int8)
        w1_np = rng.randint(-5, 5, size=(8, 4)).astype(np.int8)
        b1_np = rng.randint(-20, 20, size=4).astype(np.int32)
        w2_np = rng.randint(-5, 5, size=(4, 3)).astype(np.int8)
        b2_np = rng.randint(-20, 20, size=3).astype(np.int32)

        # Numpy reference
        h_ref = np.zeros(4, dtype=np.int32)
        for i in range(4):
            acc = np.int32(0)
            for j in range(8):
                product = np.int8(np.int8(x_np[j]) * np.int8(w1_np[j, i]))
                acc += np.int32(product)
            trunc = np.int32(np.int8(acc))
            with_bias = trunc + b1_np[i]
            if with_bias > 0:
                h_ref[i] = np.int32(np.int8(with_bias))
            else:
                h_ref[i] = 0

        logits_ref = np.zeros(3, dtype=np.int32)
        for i in range(3):
            acc = np.int32(0)
            for j in range(4):
                h_val = np.int8(h_ref[j])
                product = np.int8(h_val * np.int8(w2_np[j, i]))
                acc += np.int32(product)
            trunc = np.int32(np.int8(acc))
            logits_ref[i] = trunc + b2_np[i]

        # Simulate both kernels
        out0, _, _ = simulate_kernel(
            kernel0, {1: x_np, 2: w1_np.flatten(), 3: b1_np}
        )
        out1, _, _ = simulate_kernel(
            kernel1, {1: out0, 2: w2_np.flatten(), 3: b2_np}
        )

        np.testing.assert_array_equal(out1, logits_ref)
        # Verify we get a valid prediction
        assert out1.argmax() == logits_ref.argmax()


# ---------------------------------------------------------------------------
# MNIST-scale tests
# ---------------------------------------------------------------------------

class TestMNIST:
    @pytest.mark.slow
    def test_mnist_kernel_shapes(self):
        """MNIST model should produce 2 kernels with correct buffer shapes."""
        renderer = HDLRenderer()

        x = Tensor.empty(1, 784, dtype=dtypes.int8)
        w1 = Tensor.empty(784, 128, dtype=dtypes.int8)
        b1 = Tensor.empty(1, 128, dtype=dtypes.int32)
        w2 = Tensor.empty(128, 10, dtype=dtypes.int8)
        b2 = Tensor.empty(1, 10, dtype=dtypes.int32)

        h = ((x @ w1).cast(dtypes.int32) + b1).relu()
        h_i8 = h.cast(dtypes.int8)
        logits = (h_i8 @ w2).cast(dtypes.int32) + b2
        sched = logits.schedule()

        # Should produce 2 compute kernels
        compute_kernels = [si for si in sched if si.ast.op == Ops.SINK]
        assert len(compute_kernels) == 2

        # Kernel 0: output=128, input=784, weights=100352, bias=128
        uops0 = _get_uops(compute_kernels[0].ast, renderer)
        bufs0 = analyze_buffers(uops0)
        depths0 = {b.idx: b.depth for b in bufs0}
        assert depths0[0] == 128   # output
        assert depths0[1] == 784   # input
        assert depths0[2] == 100352  # weights (784*128)
        assert depths0[3] == 128   # bias

        # Kernel 1: output=10, input=128, weights=1280, bias=10
        uops1 = _get_uops(compute_kernels[1].ast, renderer)
        bufs1 = analyze_buffers(uops1)
        depths1 = {b.idx: b.depth for b in bufs1}
        assert depths1[0] == 10    # output
        assert depths1[1] == 128   # hidden (from kernel 0)
        assert depths1[2] == 1280  # weights (128*10)
        assert depths1[3] == 10    # bias
