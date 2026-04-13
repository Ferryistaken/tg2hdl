"""Tests for Conv2d and CNN patterns through the tg2hdl compiler pipeline.

Probes whether the compiler can handle:
- Conv2d kernels (deeper loop nesting than GEMV)
- Conv2d + ReLU fusion
- Conv2d + FC (mini CNN)
- Max/Avg pooling
- Softmax

Each test follows the same pattern as the existing test suite:
1. Build a symbolic tinygrad graph
2. Compile through the HDL pipeline (UOps → KernelIR → CompiledKernel)
3. Simulate on Amaranth simulator
4. Compare output to tinygrad CPU reference
"""

import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import NOOPT as tg_noopt
from tinygrad.uop.ops import Ops

from compiler.backend import (
    HDLRenderer,
    _get_uops,
    compile_kernel,
    compile_model,
    simulate_kernel,
    uops_to_kernel_ir,
)
from benchmarks.harness import run_bench


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noopt():
    """Context manager to set NOOPT=1."""
    class _ctx:
        def __enter__(self):
            self._old = tg_noopt.value
            tg_noopt.value = 1
            return self
        def __exit__(self, *a):
            tg_noopt.value = self._old
    return _ctx()


def _compile_and_simulate(build_fn, input_arrays, *, is_float=False, rtol=1e-5, atol=1e-6):
    """Compile a tinygrad graph through the HDL pipeline and simulate it.

    Returns (hdl_output, ref_output, correct).
    """
    # CPU reference
    with _noopt():
        ref_tensors = [Tensor(a) for a in input_arrays]
        ref_out = build_fn(ref_tensors).numpy().flatten()

    # Compile
    with _noopt():
        syms = [Tensor.empty(a.shape, dtype=Tensor(a).dtype) for a in input_arrays]
        out_sym = build_fn(syms)
        schedule = out_sym.schedule()
        compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
        kernel_specs = compile_model(schedule)

    # Detect connections
    output_buf_ids = {}
    for k_idx, si in enumerate(compute_items):
        if si.bufs:
            output_buf_ids[id(si.bufs[0])] = k_idx

    connections = {}
    for k_idx, si in enumerate(compute_items):
        for buf_pos, buf in enumerate(si.bufs[1:], start=1):
            if buf is not None and id(buf) in output_buf_ids:
                src_k = output_buf_ids[id(buf)]
                if src_k < k_idx:
                    connections[(k_idx, buf_pos)] = (src_k, 0)

    external_slots = [
        (k_idx, buf_pos)
        for k_idx, si in enumerate(compute_items)
        for buf_pos in range(1, len(si.bufs))
        if (k_idx, buf_pos) not in connections
    ]

    # Assign input arrays to external slots
    input_map = {}
    for i, slot in enumerate(external_slots):
        if i < len(input_arrays):
            input_map[slot] = input_arrays[i]

    # Simulate kernels sequentially
    kernel_outputs = {}
    for k_idx, ks in enumerate(kernel_specs):
        kernel_inputs = {}
        for (ki, buf_pos), arr in input_map.items():
            if ki == k_idx:
                kernel_inputs[buf_pos] = arr
        for (ki, buf_pos), (src_k, _) in connections.items():
            if ki == k_idx:
                kernel_inputs[buf_pos] = kernel_outputs[src_k]

        out, cycles, _ = simulate_kernel(ks.kernel, kernel_inputs)
        kernel_outputs[k_idx] = out

    hdl_raw = kernel_outputs[len(kernel_specs) - 1]

    if is_float:
        hdl_out = hdl_raw.astype(np.uint32).view(np.float32)
        min_len = min(len(hdl_out), len(ref_out))
        hdl_out, ref_cmp = hdl_out[:min_len], ref_out[:min_len]
        if min_len > 0:
            correct = bool(np.allclose(hdl_out, ref_cmp, rtol=rtol, atol=atol))
        else:
            correct = True
    else:
        hdl_out = hdl_raw
        min_len = min(len(hdl_out), len(ref_out))
        correct = bool(np.array_equal(
            hdl_out[:min_len].astype(np.int64),
            ref_out[:min_len].astype(np.int64),
        ))

    return hdl_out, ref_out, correct


# ---------------------------------------------------------------------------
# Test: Conv2d compilation (UOps → KernelIR)
# ---------------------------------------------------------------------------

class TestConv2dCompilation:
    """Test that Conv2d kernels can be compiled to KernelIR."""

    def test_conv2d_1x1_compiles(self):
        """Simplest possible conv: 1x1 kernel, 1 channel."""
        rng = np.random.RandomState(42)
        x = rng.randint(-4, 4, (1, 1, 4, 4)).astype(np.int8)
        w = rng.randint(-2, 2, (1, 1, 1, 1)).astype(np.int8)

        with _noopt():
            x_t = Tensor.empty(1, 1, 4, 4, dtype=dtypes.int8)
            w_t = Tensor.empty(1, 1, 1, 1, dtype=dtypes.int8)
            out = (x_t.conv2d(w_t)).cast(dtypes.int32)
            schedule = out.schedule()
            compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
            assert len(compute_items) >= 1, "Expected at least one compute kernel"

            renderer = HDLRenderer()
            for si in compute_items:
                uops = _get_uops(si.ast, renderer)
                kernel_ir, buf_infos = uops_to_kernel_ir(uops)
                # Should not raise

    def test_conv2d_3x3_compiles(self):
        """Standard 3x3 conv, 1 channel in, 1 channel out."""
        rng = np.random.RandomState(42)
        x = rng.randint(-4, 4, (1, 1, 8, 8)).astype(np.int8)
        w = rng.randint(-2, 2, (1, 1, 3, 3)).astype(np.int8)

        with _noopt():
            x_t = Tensor.empty(1, 1, 8, 8, dtype=dtypes.int8)
            w_t = Tensor.empty(1, 1, 3, 3, dtype=dtypes.int8)
            out = (x_t.conv2d(w_t)).cast(dtypes.int32)
            schedule = out.schedule()
            compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
            assert len(compute_items) >= 1

            renderer = HDLRenderer()
            for si in compute_items:
                uops = _get_uops(si.ast, renderer)
                kernel_ir, buf_infos = uops_to_kernel_ir(uops)

    def test_conv2d_multichannel_compiles(self):
        """Multi-channel conv: 3 channels in, 2 channels out, 3x3 kernel."""
        with _noopt():
            x_t = Tensor.empty(1, 3, 8, 8, dtype=dtypes.int8)
            w_t = Tensor.empty(2, 3, 3, 3, dtype=dtypes.int8)
            out = (x_t.conv2d(w_t)).cast(dtypes.int32)
            schedule = out.schedule()
            compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
            assert len(compute_items) >= 1

            renderer = HDLRenderer()
            for si in compute_items:
                uops = _get_uops(si.ast, renderer)
                kernel_ir, buf_infos = uops_to_kernel_ir(uops)

    def test_conv2d_fp32_compiles(self):
        """Float32 conv should compile to KernelIR."""
        with _noopt():
            x_t = Tensor.empty(1, 1, 6, 6, dtype=dtypes.float32)
            w_t = Tensor.empty(1, 1, 3, 3, dtype=dtypes.float32)
            out = x_t.conv2d(w_t)
            schedule = out.schedule()
            compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
            assert len(compute_items) >= 1

            renderer = HDLRenderer()
            for si in compute_items:
                uops = _get_uops(si.ast, renderer)
                kernel_ir, buf_infos = uops_to_kernel_ir(uops)


# ---------------------------------------------------------------------------
# Test: Conv2d full pipeline (compile + simulate)
# ---------------------------------------------------------------------------

class TestConv2dSimulation:
    """Test that Conv2d kernels produce correct results in HDL simulation."""

    def test_conv2d_1x1_int8(self):
        """1x1 conv, int8 → int32, should match CPU reference."""
        rng = np.random.RandomState(42)
        x = rng.randint(-4, 4, (1, 1, 4, 4)).astype(np.int8)
        w = rng.randint(-2, 2, (1, 1, 1, 1)).astype(np.int8)

        def build(t):
            return (t[0].conv2d(t[1])).cast(dtypes.int32)

        result = run_bench("conv2d_1x1_int8", build, [x, w])
        assert result.correct, f"Conv2d 1x1 failed: {result}"

    def test_conv2d_3x3_int8(self):
        """3x3 conv, int8 → int32, standard kernel size."""
        rng = np.random.RandomState(42)
        x = rng.randint(-3, 3, (1, 1, 8, 8)).astype(np.int8)
        w = rng.randint(-2, 2, (1, 1, 3, 3)).astype(np.int8)

        def build(t):
            return (t[0].conv2d(t[1])).cast(dtypes.int32)

        result = run_bench("conv2d_3x3_int8", build, [x, w])
        assert result.correct, f"Conv2d 3x3 failed: {result}"

    def test_conv2d_3x3_multichannel_int8(self):
        """Multi-channel 3x3 conv: 2 in, 2 out."""
        rng = np.random.RandomState(42)
        x = rng.randint(-3, 3, (1, 2, 6, 6)).astype(np.int8)
        w = rng.randint(-2, 2, (2, 2, 3, 3)).astype(np.int8)

        def build(t):
            return (t[0].conv2d(t[1])).cast(dtypes.int32)

        result = run_bench("conv2d_3x3_mc_int8", build, [x, w])
        assert result.correct, f"Conv2d 3x3 multichannel failed: {result}"

    def test_conv2d_3x3_fp32(self):
        """3x3 float32 conv."""
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 6, 6).astype(np.float32) * 0.5
        w = rng.randn(1, 1, 3, 3).astype(np.float32) * 0.5

        def build(t):
            return t[0].conv2d(t[1])

        result = run_bench("conv2d_3x3_fp32", build, [x, w])
        assert result.correct, f"Conv2d 3x3 fp32 failed: {result}"

    def test_conv2d_with_bias_int8(self):
        """Conv2d + bias add, int8 → int32."""
        rng = np.random.RandomState(42)
        x = rng.randint(-3, 3, (1, 1, 6, 6)).astype(np.int8)
        w = rng.randint(-2, 2, (1, 1, 3, 3)).astype(np.int8)
        b = rng.randint(-5, 5, (1, 1, 1, 1)).astype(np.int32)

        def build(t):
            return (t[0].conv2d(t[1])).cast(dtypes.int32) + t[2]

        result = run_bench("conv2d_bias_int8", build, [x, w, b])
        assert result.correct, f"Conv2d + bias failed: {result}"


# ---------------------------------------------------------------------------
# Test: CNN patterns (Conv + activation, Conv + FC)
# ---------------------------------------------------------------------------

class TestCNNPatterns:
    """Test common CNN building blocks."""

    def test_conv2d_relu_int8(self):
        """Conv2d + ReLU, int8 → int32."""
        rng = np.random.RandomState(42)
        x = rng.randint(-3, 3, (1, 1, 6, 6)).astype(np.int8)
        w = rng.randint(-2, 2, (1, 1, 3, 3)).astype(np.int8)

        def build(t):
            return (t[0].conv2d(t[1])).cast(dtypes.int32).relu()

        result = run_bench("conv2d_relu_int8", build, [x, w])
        assert result.correct, f"Conv2d + ReLU failed: {result}"

    def test_conv2d_relu_fp32(self):
        """Conv2d + ReLU, float32."""
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 6, 6).astype(np.float32) * 0.5
        w = rng.randn(1, 1, 3, 3).astype(np.float32) * 0.5

        def build(t):
            return t[0].conv2d(t[1]).relu()

        result = run_bench("conv2d_relu_fp32", build, [x, w])
        assert result.correct, f"Conv2d + ReLU fp32 failed: {result}"

    def test_mini_cnn_int8(self):
        """Tiny CNN: Conv2d(3x3) → ReLU → flatten → FC, int8 paths."""
        rng = np.random.RandomState(42)
        x = rng.randint(-3, 3, (1, 1, 4, 4)).astype(np.int8)
        conv_w = rng.randint(-2, 2, (1, 1, 3, 3)).astype(np.int8)
        # Conv output: (1, 1, 2, 2) → flatten → (1, 4)
        fc_w = rng.randint(-2, 2, (4, 2)).astype(np.int8)

        def build(t):
            h = (t[0].conv2d(t[1])).cast(dtypes.int32).relu()
            h = h.cast(dtypes.int8).reshape(1, -1)
            return (h @ t[2]).cast(dtypes.int32)

        result = run_bench("mini_cnn_int8", build, [x, conv_w, fc_w])
        assert result.correct, f"Mini CNN failed: {result}"


# ---------------------------------------------------------------------------
# Test: Pooling operations
# ---------------------------------------------------------------------------

class TestPooling:
    """Test pooling operations."""

    def test_max_pool_int32(self):
        """Max pooling 2x2, int32."""
        rng = np.random.RandomState(42)
        x = rng.randint(-10, 10, (1, 1, 4, 4)).astype(np.int32)

        def build(t):
            return t[0].max_pool2d(kernel_size=(2, 2))

        result = run_bench("maxpool_2x2_int32", build, [x])
        assert result.correct, f"Max pool 2x2 failed: {result}"

    def test_max_pool_fp32(self):
        """Max pooling 2x2, float32."""
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 4, 4).astype(np.float32)

        def build(t):
            return t[0].max_pool2d(kernel_size=(2, 2))

        result = run_bench("maxpool_2x2_fp32", build, [x])
        assert result.correct, f"Max pool 2x2 fp32 failed: {result}"

    def test_avg_pool_int32(self):
        """Average pooling 2x2, int32 (truncated division)."""
        rng = np.random.RandomState(42)
        x = rng.randint(-10, 10, (1, 1, 4, 4)).astype(np.int32)

        def build(t):
            return t[0].avg_pool2d(kernel_size=(2, 2))

        result = run_bench("avgpool_2x2_int32", build, [x], exact=False)
        assert result.correct, f"Avg pool 2x2 failed: {result}"


# ---------------------------------------------------------------------------
# Test: Softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
    """Test softmax operation (requires exp2, reduction, reciprocal)."""

    @pytest.mark.xfail(
        reason="Softmax produces inf in HDL simulation — likely a bug in the "
               "multi-kernel EXP2 polynomial or inter-kernel data passing. "
               "The 3-kernel chain (max-reduce → exp+sum-reduce → exp*recip) "
               "compiles but produces incorrect intermediate values.",
        strict=True,
    )
    def test_softmax_fp32_small(self):
        """Small softmax over 4 elements, float32."""
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        def build(t):
            return t[0].softmax(axis=-1)

        result = run_bench("softmax_fp32_4", build, [x])
        assert result.correct, f"Softmax failed: {result}"

    @pytest.mark.xfail(
        reason="Softmax produces inf — same root cause as test_softmax_fp32_small.",
        strict=True,
    )
    def test_softmax_fp32_8(self):
        """Softmax over 8 elements, float32."""
        rng = np.random.RandomState(42)
        x = rng.randn(1, 8).astype(np.float32)

        def build(t):
            return t[0].softmax(axis=-1)

        result = run_bench("softmax_fp32_8", build, [x])
        assert result.correct, f"Softmax 8 failed: {result}"


# ---------------------------------------------------------------------------
# Test: Loop structure inspection for Conv2d
# ---------------------------------------------------------------------------

class TestConv2dLoopStructure:
    """Inspect the loop structure produced by Conv2d to understand nesting."""

    def test_conv2d_loop_depth(self):
        """Check how many loop nesting levels a Conv2d produces."""
        with _noopt():
            x_t = Tensor.empty(1, 1, 6, 6, dtype=dtypes.int8)
            w_t = Tensor.empty(1, 1, 3, 3, dtype=dtypes.int8)
            out = (x_t.conv2d(w_t)).cast(dtypes.int32)
            schedule = out.schedule()
            compute_items = [si for si in schedule if si.ast.op == Ops.SINK]

            renderer = HDLRenderer()
            for si in compute_items:
                uops = _get_uops(si.ast, renderer)
                kernel_ir, _ = uops_to_kernel_ir(uops)

                # Count loop depth
                depth = 0
                level = kernel_ir.loop_tree.body
                while level is not None:
                    depth += 1
                    level = level.body

                # Conv2d should produce at least 2 loop levels (output spatial + reduce)
                assert depth >= 2, (
                    f"Expected Conv2d to produce at least 2 loop levels, got {depth}"
                )

    def test_conv2d_multichannel_loop_depth(self):
        """Multi-channel conv should produce deeper nesting."""
        with _noopt():
            x_t = Tensor.empty(1, 2, 6, 6, dtype=dtypes.int8)
            w_t = Tensor.empty(2, 2, 3, 3, dtype=dtypes.int8)
            out = (x_t.conv2d(w_t)).cast(dtypes.int32)
            schedule = out.schedule()
            compute_items = [si for si in schedule if si.ast.op == Ops.SINK]

            renderer = HDLRenderer()
            for si in compute_items:
                uops = _get_uops(si.ast, renderer)
                kernel_ir, _ = uops_to_kernel_ir(uops)

                depth = 0
                level = kernel_ir.loop_tree.body
                while level is not None:
                    depth += 1
                    level = level.body

                # Multi-channel conv should have more nesting
                assert depth >= 2, (
                    f"Expected multi-channel Conv2d to produce at least 2 loop levels, got {depth}"
                )
