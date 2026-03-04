"""Benchmark harness: compare any tinygrad computation against HDL simulation.

Supported dtype paths
---------------------
Integer dtypes (int8, int16, int32)
    Full Amaranth simulation.  Output is bit-exact.  Correctness is verified
    by comparing the simulated HDL output to the tinygrad CPU reference.

Float32
    Full Amaranth simulation using IEEE 754 hardware modules (FP32Add, FP32Mul,
    FP32Cmp).  Results are compared to the tinygrad CPU reference with
    ``rtol=1e-5, atol=1e-6``.  The same elaboratable hardware is used for
    simulation and synthesis — there is no software float fallback.
    ``result.float_path`` is always ``False``.

    Known limitations: subnormals flush to zero; rounding is truncation
    (round-toward-zero) rather than IEEE default round-to-nearest-even.

Float16 / BFloat16
    No dedicated arithmetic units — compile error at any arithmetic op.

Multi-kernel connection detection
----------------------------------
Buffer object identity (``id(si.bufs[0]) == id(si.bufs[j])``) detects when
one kernel's output feeds into another kernel's input.  ``input_arrays`` are
assigned to external (non-connected) input buffer slots in natural schedule
order.
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import NOOPT as tg_noopt
from tinygrad.uop.ops import Ops

from compiler.backend import (
    CompileOptions,
    compile_model,
    simulate_kernel,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    """Result of a single benchmark run."""
    name: str
    correct: bool           # True if HDL output matches reference (or float path)
    max_abs_error: float    # 0 for exact integer, may be non-zero for float/tolerance
    hdl_cycles: int         # total hardware cycles across all kernels
    sim_wall_s: float       # simulation wall-clock seconds (0.0 for float path)
    tg_wall: float          # tinygrad optimized CPU reference wall-clock seconds
    tg_wall_noopt: float    # tinygrad NOOPT=1 wall-clock seconds
    float_path: bool        # True when float-mode path was used
    output_hdl: np.ndarray
    output_ref: np.ndarray

    def __str__(self):
        mode = "float-analytical" if self.float_path else "hdl-sim"
        status = "PASS" if self.correct else "FAIL"
        return (
            f"[{status}] {self.name} [{mode}]: "
            f"err={self.max_abs_error:.4g} cycles={self.hdl_cycles} "
            f"sim={self.sim_wall_s:.3f}s tg={self.tg_wall:.4f}s "
            f"tg_noopt={self.tg_wall_noopt:.4f}s"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_float(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.floating)


def _detect_connections(compute_items):
    """Return (connections_dict, external_slots) from a list of ExecItems."""
    output_buf_ids: dict[int, int] = {}
    for k_idx, si in enumerate(compute_items):
        if si.bufs:
            output_buf_ids[id(si.bufs[0])] = k_idx

    connections: dict[tuple, tuple] = {}
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
    return connections, external_slots


@contextmanager
def _noopt_scope(value=1):
    old = tg_noopt.value
    tg_noopt.value = value
    try:
        yield
    finally:
        tg_noopt.value = old


# ---------------------------------------------------------------------------
# run_bench
# ---------------------------------------------------------------------------

def run_bench(name: str, build_fn, input_arrays: list, exact: bool = True, *, unroll_loop: int = 1) -> BenchResult:
    """Run *build_fn* on tinygrad CPU and HDL simulation, compare outputs.

    Parameters
    ----------
    name : str
        Human-readable benchmark name.
    build_fn : callable
        ``list[Tensor] → Tensor``.  Must work with both real and empty tensors.
    input_arrays : list[np.ndarray]
        Input data in the same order as the tensors ``build_fn`` receives.
        Dtypes may be int8, int16, int32, float16, or float32.
    exact : bool
        True  → require bit-exact match (integer path).
        False → allow ±1 absolute error (use for expected truncation effects).

    Returns
    -------
    BenchResult
    """
    try:
        # ------------------------------------------------------------------
        # 1. tinygrad CPU reference
        # ------------------------------------------------------------------
        # warm up both execution modes outside timing to avoid one-time compile skew
        _ = build_fn([Tensor(a) for a in input_arrays]).numpy().flatten()
        with _noopt_scope(1):
            _ = build_fn([Tensor(a) for a in input_arrays]).numpy().flatten()

        with _noopt_scope(1):
            t1 = time.perf_counter()
            ref_noopt = build_fn([Tensor(a) for a in input_arrays]).numpy().flatten()
            tg_wall_noopt = time.perf_counter() - t1

        t0 = time.perf_counter()
        ref_tensors = [Tensor(a) for a in input_arrays]
        ref_out = build_fn(ref_tensors).numpy().flatten()
        tg_wall = time.perf_counter() - t0

        is_float_output = np.issubdtype(ref_out.dtype, np.floating)

    # ------------------------------------------------------------------
    # 2. Build symbolic graph and compile
    # ------------------------------------------------------------------
        syms = [Tensor.empty(a.shape, dtype=Tensor(a).dtype) for a in input_arrays]
        out_sym = build_fn(syms)
        with _noopt_scope(1):
            schedule = out_sym.schedule()
            compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
            kernel_specs = compile_model(schedule, options=CompileOptions(unroll_loop=unroll_loop))

    # ------------------------------------------------------------------
    # 3. Amaranth simulation (integer and float32 paths both go through here)
    # ------------------------------------------------------------------
        connections, external_slots = _detect_connections(compute_items)

        input_map: dict[tuple, np.ndarray] = {}
        for i, slot in enumerate(external_slots):
            if i < len(input_arrays):
                input_map[slot] = input_arrays[i]

        kernel_outputs: dict[int, np.ndarray] = {}
        total_cycles = 0
        t0 = time.perf_counter()

        for k_idx, ks in enumerate(kernel_specs):
            kernel_inputs: dict[int, np.ndarray] = {}
            for (ki, buf_pos), arr in input_map.items():
                if ki == k_idx:
                    kernel_inputs[buf_pos] = arr
            for (ki, buf_pos), (src_k, _) in connections.items():
                if ki == k_idx:
                    kernel_inputs[buf_pos] = kernel_outputs[src_k]

            out, cycles, _ = simulate_kernel(ks.kernel, kernel_inputs)
            kernel_outputs[k_idx] = out
            total_cycles += cycles

        sim_wall = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 4. Compare — dtype-aware
    # ------------------------------------------------------------------
        hdl_raw = kernel_outputs[len(kernel_specs) - 1]

        if is_float_output:
            # simulate_kernel returns raw uint32 bit patterns as int32.
            # Re-interpret as float32 for comparison.
            hdl_out = hdl_raw.astype(np.uint32).view(np.float32)
            min_len = min(len(hdl_out), len(ref_out))
            hdl_out, ref_cmp = hdl_out[:min_len], ref_out[:min_len]

            # Use relative + absolute tolerance (truncation rounding mode)
            if min_len > 0:
                max_err = float(np.max(np.abs(hdl_out.astype(np.float64)
                                              - ref_cmp.astype(np.float64))))
                # tolerate 1 ULP of relative error from truncation rounding
                correct = bool(np.allclose(hdl_out, ref_cmp, rtol=1e-5, atol=1e-6))
            else:
                max_err, correct = 0.0, True
        else:
            hdl_out = hdl_raw
            ref_flat = ref_out.astype(np.int64)
            hdl_flat = hdl_out.astype(np.int64)
            min_len = min(len(ref_flat), len(hdl_flat))
            ref_flat, hdl_flat = ref_flat[:min_len], hdl_flat[:min_len]

            max_err_int = int(np.max(np.abs(hdl_flat - ref_flat))) if min_len > 0 else 0
            max_err = float(max_err_int)
            correct = (np.array_equal(hdl_flat, ref_flat) if exact
                       else max_err_int <= 1)

        return BenchResult(
            name=name,
            correct=correct,
            max_abs_error=max_err,
            hdl_cycles=total_cycles,
            sim_wall_s=sim_wall,
            tg_wall=tg_wall,
            tg_wall_noopt=tg_wall_noopt,
            float_path=False,
            output_hdl=hdl_out,
            output_ref=ref_out,
        )
    finally:
        tg_noopt.value = 1


# ---------------------------------------------------------------------------
# CLI: python -m benchmarks.harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    os.environ.setdefault("DEBUG", "0")

    from tinygrad import dtypes

    rng = np.random.RandomState(0)

    cases = [
        (
            "relu_int32_4",
            lambda t: t[0].relu(),
            [rng.randint(-10, 10, 4).astype(np.int32)],
        ),
        (
            "add_int32_4",
            lambda t: t[0] + t[1],
            [rng.randint(-5, 5, 4).astype(np.int32),
             rng.randint(-5, 5, 4).astype(np.int32)],
        ),
        (
            "matmul_int8_1x4_4x3",
            lambda t: (t[0] @ t[1]).cast(dtypes.int32),
            [rng.randint(-4, 4, (1, 4)).astype(np.int8),
             rng.randint(-4, 4, (4, 3)).astype(np.int8)],
        ),
        (
            "relu_float32_8",
            lambda t: t[0].relu(),
            [rng.randn(8).astype(np.float32)],
        ),
    ]

    all_pass = True
    for case_name, fn, arrays in cases:
        r = run_bench(case_name, fn, arrays)
        print(r)
        all_pass = all_pass and r.correct

    sys.exit(0 if all_pass else 1)
