"""Tests for TopModule and compile_top_module.

Verifies:
  - TopModule can be constructed from compiled kernels
  - simulate_top produces the same output as chained simulate_kernel
  - compile_top_module auto-detects inter-kernel connections
  - done signal fires after computation completes
"""

import os

os.environ["NOOPT"] = "1"
os.environ["DEBUG"] = "0"

import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops

from compiler import (
    HDLRenderer,
    compile_kernel,
    compile_model,
    compile_top_module,
    simulate_kernel,
)
from compiler.backend import _get_uops
from compiler.top_module import TopModule, simulate_top


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp_small():
    """Return (schedule, numpy_weights, buf_infos_k0, buf_infos_k1)
    for a small 2-layer MLP: (1,4)→(1,3)→(1,2).
    """
    rng = np.random.RandomState(99)
    x_np  = rng.randint(-4, 4, (1, 4)).astype(np.int8)
    w1_np = rng.randint(-3, 3, (4, 3)).astype(np.int8)
    b1_np = rng.randint(-10, 10, (1, 3)).astype(np.int32)
    w2_np = rng.randint(-3, 3, (3, 2)).astype(np.int8)
    b2_np = rng.randint(-5, 5, (1, 2)).astype(np.int32)

    x_sym  = Tensor.empty(1, 4, dtype=dtypes.int8)
    w1_sym = Tensor.empty(4, 3, dtype=dtypes.int8)
    b1_sym = Tensor.empty(1, 3, dtype=dtypes.int32)
    w2_sym = Tensor.empty(3, 2, dtype=dtypes.int8)
    b2_sym = Tensor.empty(1, 2, dtype=dtypes.int32)

    h = ((x_sym @ w1_sym).cast(dtypes.int32) + b1_sym).relu()
    logits = (h.cast(dtypes.int8) @ w2_sym).cast(dtypes.int32) + b2_sym
    schedule = logits.schedule()

    data = {
        "x": x_np,
        "w1": w1_np,
        "b1": b1_np,
        "w2": w2_np,
        "b2": b2_np,
    }
    return schedule, data


def _ref_mlp_small(data):
    """Numpy reference matching UOp INT8 truncation semantics."""
    x_np  = data["x"].flatten()
    w1_np = data["w1"]
    b1_np = data["b1"].flatten()
    w2_np = data["w2"]
    b2_np = data["b2"].flatten()

    h_ref = np.zeros(3, dtype=np.int32)
    for i in range(3):
        acc = np.int32(0)
        for j in range(4):
            acc += np.int32(np.int8(x_np[j] * w1_np[j, i]))
        trunc = np.int32(np.int8(acc))
        v = trunc + b1_np[i]
        h_ref[i] = np.int32(np.int8(v)) if v > 0 else 0

    logits_ref = np.zeros(2, dtype=np.int32)
    for i in range(2):
        acc = np.int32(0)
        for j in range(3):
            acc += np.int32(np.int8(np.int8(h_ref[j]) * w2_np[j, i]))
        logits_ref[i] = np.int32(np.int8(acc)) + b2_np[i]

    return logits_ref


def _compile_single_kernel(build_fn, arrays):
    syms = [Tensor.empty(a.shape, dtype=Tensor(a).dtype) for a in arrays]
    expr = build_fn(syms)
    schedule = expr.schedule()
    compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
    assert len(compute_items) == 1
    uops = _get_uops(compute_items[0].ast, HDLRenderer())
    return compile_kernel(uops)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestTopModuleConstruction:
    def test_build_from_two_kernels(self):
        schedule, data = _make_mlp_small()
        top, connections, kernel_specs = compile_top_module(schedule)

        assert isinstance(top, TopModule)
        assert len(top.kernels) == 2
        assert len(connections) >= 1
        assert hasattr(top, "start")
        assert hasattr(top, "done")
        assert hasattr(top, "output_rport")

    def test_ext_write_ports_exposed(self):
        """External write ports should be present for all non-connected inputs."""
        schedule, data = _make_mlp_small()
        top, connections, _ = compile_top_module(schedule)

        # Connected buffers (copy targets) should NOT be in ext_write_ports
        internal = {(dst_k, dst_buf) for _, _, dst_k, dst_buf in connections}
        for key in top.ext_write_ports:
            assert key not in internal, f"{key} is internal but exposed as ext port"

    def test_connection_detected(self):
        """compile_top_module must detect at least one kernel-to-kernel connection."""
        schedule, _ = _make_mlp_small()
        _, connections, _ = compile_top_module(schedule)
        assert len(connections) > 0

    def test_output_rport_exists(self):
        schedule, _ = _make_mlp_small()
        top, _, _ = compile_top_module(schedule)
        assert "raddr" in top.output_rport
        assert "rdata" in top.output_rport


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestTopModuleSimulation:
    def test_simulate_top_matches_chained_kernel_sim(self):
        """simulate_top output must equal chained simulate_kernel output."""
        schedule, data = _make_mlp_small()
        top, connections, kernel_specs = compile_top_module(schedule)

        # --- Reference via chained simulate_kernel (same as existing tests) ---
        kernel0 = kernel_specs[0].kernel
        kernel1 = kernel_specs[1].kernel

        out0, _, _ = simulate_kernel(
            kernel0, {1: data["x"].flatten(), 2: data["w1"].flatten(), 3: data["b1"].flatten()}
        )
        out1_ref, _, _ = simulate_kernel(
            kernel1, {1: out0, 2: data["w2"].flatten(), 3: data["b2"].flatten()}
        )

        # --- Simulate via TopModule ---
        # Build input_data: (k_idx, buf_idx) → np.ndarray
        # K0: buf1=x, buf2=w1, buf3=b1
        # K1: buf2=w2, buf3=b2  (buf1 is connected from K0, not exposed)
        input_data = {}
        for (k_idx, buf_idx) in top.ext_write_ports:
            if k_idx == 0:
                arr_map = {1: data["x"].flatten(), 2: data["w1"].flatten(), 3: data["b1"].flatten()}
                input_data[(k_idx, buf_idx)] = arr_map[buf_idx]
            else:
                arr_map = {2: data["w2"].flatten(), 3: data["b2"].flatten()}
                if buf_idx in arr_map:
                    input_data[(k_idx, buf_idx)] = arr_map[buf_idx]

        out_top, _, _ = simulate_top(top, input_data)

        np.testing.assert_array_equal(
            out_top, out1_ref,
            err_msg="TopModule output differs from chained simulate_kernel"
        )

    def test_simulate_top_matches_numpy_reference(self):
        """simulate_top must produce bit-exact numpy reference output."""
        schedule, data = _make_mlp_small()
        top, connections, kernel_specs = compile_top_module(schedule)

        # Build input_data dict
        input_data = {}
        for (k_idx, buf_idx) in top.ext_write_ports:
            if k_idx == 0:
                arr_map = {1: data["x"].flatten(), 2: data["w1"].flatten(), 3: data["b1"].flatten()}
                input_data[(k_idx, buf_idx)] = arr_map[buf_idx]
            else:
                arr_map = {2: data["w2"].flatten(), 3: data["b2"].flatten()}
                if buf_idx in arr_map:
                    input_data[(k_idx, buf_idx)] = arr_map[buf_idx]

        out_top, _, _ = simulate_top(top, input_data)
        expected = _ref_mlp_small(data)

        np.testing.assert_array_equal(out_top, expected)

    def test_simulate_top_done_fires(self):
        """simulate_top must report non-zero cycles (i.e. done was observed)."""
        schedule, data = _make_mlp_small()
        top, _, _ = compile_top_module(schedule)

        input_data = {}
        for (k_idx, buf_idx) in top.ext_write_ports:
            if k_idx == 0:
                arr_map = {1: data["x"].flatten(), 2: data["w1"].flatten(), 3: data["b1"].flatten()}
                input_data[(k_idx, buf_idx)] = arr_map[buf_idx]
            else:
                arr_map = {2: data["w2"].flatten(), 3: data["b2"].flatten()}
                if buf_idx in arr_map:
                    input_data[(k_idx, buf_idx)] = arr_map[buf_idx]

        _, cycle_counts, wall = simulate_top(top, input_data)
        assert cycle_counts["compute"] > 0, "No cycles recorded — done never observed"
        assert wall > 0.0

    def test_manual_non_adjacent_dependency(self):
        """TopModule must support copying producer output to a non-adjacent later kernel."""
        x = np.array([1, 2, 3, 4], dtype=np.int32)
        y = np.array([9, 9, 9, 9], dtype=np.int32)

        k0 = _compile_single_kernel(lambda t: t[0] + 1, [x])
        k1 = _compile_single_kernel(lambda t: t[0] + 2, [y])
        k2 = _compile_single_kernel(lambda t: t[0] * 3, [x])

        top = TopModule(
            [k0, k1, k2],
            connections=[(0, 0, 2, 1)],
            buf_depths={
                (0, 0): 4, (0, 1): 4,
                (1, 0): 4, (1, 1): 4,
                (2, 0): 4, (2, 1): 4,
            },
        )
        out, _, _ = simulate_top(top, {(0, 1): x, (1, 1): y})
        expected = ((x + 1) * 3).astype(np.int32)
        np.testing.assert_array_equal(out, expected)

    def test_manual_fanout_dependency(self):
        """TopModule must support one producer feeding multiple later kernels."""
        x = np.array([2, 4, 6, 8], dtype=np.int32)

        k0 = _compile_single_kernel(lambda t: t[0] + 1, [x])
        k1 = _compile_single_kernel(lambda t: t[0] * 2, [x])
        k2 = _compile_single_kernel(lambda t: t[0] * 3, [x])

        top = TopModule(
            [k0, k1, k2],
            connections=[(0, 0, 1, 1), (0, 0, 2, 1)],
            buf_depths={
                (0, 0): 4, (0, 1): 4,
                (1, 0): 4, (1, 1): 4,
                (2, 0): 4, (2, 1): 4,
            },
        )
        out, _, _ = simulate_top(top, {(0, 1): x})
        expected = ((x + 1) * 3).astype(np.int32)
        np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# compile_top_module connection-detection test
# ---------------------------------------------------------------------------

class TestCompileTopModule:
    def test_single_kernel_no_connections(self):
        """A single-kernel graph should produce an empty connections list."""
        x = Tensor.empty(1, 4, dtype=dtypes.int8)
        w = Tensor.empty(4, 3, dtype=dtypes.int8)
        out = (x @ w).cast(dtypes.int32)
        schedule = out.schedule()

        top, connections, kernel_specs = compile_top_module(schedule)
        assert len(connections) == 0
        assert len(top.kernels) == 1

    def test_two_kernel_connection_src_dst(self):
        """Connection must go from kernel 0 to kernel 1."""
        schedule, _ = _make_mlp_small()
        _, connections, _ = compile_top_module(schedule)

        for src_k, src_buf, dst_k, dst_buf in connections:
            assert src_k < dst_k, "All connections must be forward"
            assert src_buf == 0, "Source of copy is always the output buffer"
