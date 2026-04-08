from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops

from benchmarks.harness import BenchResult, run_bench
from compiler import HDLRenderer, compile_kernel, simulate_top
from compiler.backend import _get_uops
from compiler.top_module import TopModule


Status = str


@dataclass(frozen=True)
class ScoreboardCase:
    name: str
    tier: int
    category: str
    feature: str
    description: str
    probe: Callable[[], "ScoreboardResult"]


@dataclass
class ScoreboardResult:
    name: str
    tier: int
    category: str
    feature: str
    status: Status
    description: str
    correctness: bool | None = None
    cycles: int | None = None
    sim_wall_s: float | None = None
    tg_wall_s: float | None = None
    max_abs_error: float | None = None
    detail: str = ""
    error_type: str | None = None


def _classify_exception(exc: Exception) -> tuple[Status, str]:
    msg = str(exc).lower()
    if isinstance(exc, NotImplementedError):
        return "unsupported", type(exc).__name__
    if "unsupported" in msg or "not implemented" in msg or "not yet supported" in msg:
        return "unsupported", type(exc).__name__
    return "error", type(exc).__name__


def _from_bench(case: ScoreboardCase, bench: BenchResult) -> ScoreboardResult:
    return ScoreboardResult(
        name=case.name,
        tier=case.tier,
        category=case.category,
        feature=case.feature,
        status="pass" if bench.correct else "fail",
        description=case.description,
        correctness=bench.correct,
        cycles=bench.hdl_cycles,
        sim_wall_s=bench.sim_wall_s,
        tg_wall_s=bench.tg_wall,
        max_abs_error=bench.max_abs_error,
        detail=str(bench),
    )


def _run_case(case: ScoreboardCase) -> ScoreboardResult:
    try:
        return case.probe()
    except Exception as exc:
        status, error_type = _classify_exception(exc)
        return ScoreboardResult(
            name=case.name,
            tier=case.tier,
            category=case.category,
            feature=case.feature,
            status=status,
            description=case.description,
            detail=str(exc),
            error_type=error_type,
        )


def _probe_bench(
    case: ScoreboardCase,
    build_fn: Callable,
    input_arrays: list[np.ndarray],
    *,
    exact: bool = True,
    unroll_factor: int = 1,
    reduce_unroll_factor: int = 1,
) -> ScoreboardResult:
    bench = run_bench(
        case.name,
        build_fn,
        input_arrays,
        exact=exact,
        unroll_factor=unroll_factor,
        reduce_unroll_factor=reduce_unroll_factor,
    )
    return _from_bench(case, bench)


def _compile_single_kernel(build_fn: Callable, input_arrays: list[np.ndarray]):
    syms = [Tensor.empty(a.shape, dtype=Tensor(a).dtype) for a in input_arrays]
    expr = build_fn(syms)
    schedule = expr.schedule()
    compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
    if len(compute_items) != 1:
        raise RuntimeError(f"Expected exactly one compute kernel, got {len(compute_items)}")
    uops = _get_uops(compute_items[0].ast, HDLRenderer())
    return compile_kernel(uops)


def _make_buf_depths(kernels: list) -> dict[tuple[int, int], int]:
    buf_depths: dict[tuple[int, int], int] = {}
    for k_idx, kernel in enumerate(kernels):
        for info in kernel.buf_infos:
            buf_depths[(k_idx, info["idx"])] = info["depth"]
    return buf_depths


def _probe_manual_top(
    case: ScoreboardCase,
    *,
    kernels: list,
    connections: list[tuple[int, int, int, int]],
    input_data: dict[tuple[int, int], np.ndarray],
    expected: np.ndarray,
    is_float: bool = False,
) -> ScoreboardResult:
    top = TopModule(kernels, connections, _make_buf_depths(kernels))
    out, cycles, wall = simulate_top(top, input_data)
    if is_float:
        got = out.astype(np.uint32).view(np.float32)
        expected_cmp = expected.astype(np.float32)
        correct = bool(np.allclose(got, expected_cmp, rtol=1e-5, atol=1e-6))
        max_abs_error = float(np.max(np.abs(got.astype(np.float64) - expected_cmp.astype(np.float64))))
    else:
        got = out.astype(np.int64)
        expected_cmp = expected.astype(np.int64)
        correct = bool(np.array_equal(got, expected_cmp))
        max_abs_error = float(np.max(np.abs(got - expected_cmp)))
    return ScoreboardResult(
        name=case.name,
        tier=case.tier,
        category=case.category,
        feature=case.feature,
        status="pass" if correct else "fail",
        description=case.description,
        correctness=correct,
        cycles=cycles,
        sim_wall_s=wall,
        max_abs_error=max_abs_error,
        detail=f"got={got.tolist()} expected={expected_cmp.tolist()}",
    )


def _score_add_int32(case: ScoreboardCase) -> ScoreboardResult:
    a = np.array([7, -3, 5, 11], dtype=np.int32)
    b = np.array([2, 4, -8, 1], dtype=np.int32)
    return _probe_bench(case, lambda t: t[0] + t[1], [a, b])


def _score_relu_int32(case: ScoreboardCase) -> ScoreboardResult:
    a = np.array([-5, -1, 0, 2, 9, -7, 3, 4], dtype=np.int32)
    return _probe_bench(case, lambda t: t[0].relu(), [a])


def _score_gemv_int8(case: ScoreboardCase) -> ScoreboardResult:
    rng = np.random.RandomState(17)
    x = rng.randint(-4, 4, (1, 8)).astype(np.int8)
    w = rng.randint(-4, 4, (8, 16)).astype(np.int8)
    return _probe_bench(case, lambda t: (t[0] @ t[1]).cast(dtypes.int32), [x, w])


def _score_gemv_fp32(case: ScoreboardCase) -> ScoreboardResult:
    a = np.array([[1.0, 2.0, -1.5, 0.5]], dtype=np.float32)
    b = np.array(
        [
            [0.5, 1.0, -2.0],
            [1.5, -0.5, 0.25],
            [2.0, 0.0, 1.0],
            [-1.0, 3.0, 2.0],
        ],
        dtype=np.float32,
    )
    return _probe_bench(case, lambda t: t[0] @ t[1], [a, b])


def _score_mod_int32(case: ScoreboardCase) -> ScoreboardResult:
    a = np.array([7, 8, 9, 10], dtype=np.int32)
    b = np.array([3, 5, 2, 6], dtype=np.int32)
    return _probe_bench(case, lambda t: t[0] % t[1], [a, b])


def _score_cmp_eq_int32(case: ScoreboardCase) -> ScoreboardResult:
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    b = np.array([1, 0, 3, 7, 5], dtype=np.int32)
    return _probe_bench(case, lambda t: (t[0] == t[1]).cast(dtypes.int32), [a, b])


def _score_three_layer_chain(case: ScoreboardCase) -> ScoreboardResult:
    rng = np.random.RandomState(23)
    x = rng.randint(-4, 4, (1, 6)).astype(np.int8)
    w1 = rng.randint(-3, 3, (6, 5)).astype(np.int8)
    b1 = rng.randint(-10, 10, (1, 5)).astype(np.int32)
    w2 = rng.randint(-3, 3, (5, 4)).astype(np.int8)
    b2 = rng.randint(-8, 8, (1, 4)).astype(np.int32)
    w3 = rng.randint(-3, 3, (4, 3)).astype(np.int8)
    b3 = rng.randint(-6, 6, (1, 3)).astype(np.int32)

    def build(t):
        h1 = ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu()
        h2 = ((h1.cast(dtypes.int8) @ t[3]).cast(dtypes.int32) + t[4]).relu()
        return (h2.cast(dtypes.int8) @ t[5]).cast(dtypes.int32) + t[6]

    return _probe_bench(case, build, [x, w1, b1, w2, b2, w3, b3])


def _score_reduce_unroll(case: ScoreboardCase) -> ScoreboardResult:
    rng = np.random.RandomState(29)
    x = rng.randint(-4, 4, (1, 8)).astype(np.int8)
    w = rng.randint(-4, 4, (8, 16)).astype(np.int8)
    return _probe_bench(
        case,
        lambda t: (t[0] @ t[1]).cast(dtypes.int32),
        [x, w],
        reduce_unroll_factor=2,
    )


def _score_top_adjacent_chain(case: ScoreboardCase) -> ScoreboardResult:
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    k0 = _compile_single_kernel(lambda t: t[0] + 1, [x])
    k1 = _compile_single_kernel(lambda t: t[0] * 2, [x])
    expected = (x + 1) * 2
    return _probe_manual_top(
        case,
        kernels=[k0, k1],
        connections=[(0, 0, 1, 1)],
        input_data={(0, 1): x},
        expected=expected,
    )


def _score_top_non_adjacent_dependency(case: ScoreboardCase) -> ScoreboardResult:
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([9, 9, 9, 9], dtype=np.int32)
    k0 = _compile_single_kernel(lambda t: t[0] + 1, [x])
    k1 = _compile_single_kernel(lambda t: t[0] + 2, [y])
    k2 = _compile_single_kernel(lambda t: t[0] * 3, [x])
    expected = (x + 1) * 3
    return _probe_manual_top(
        case,
        kernels=[k0, k1, k2],
        connections=[(0, 0, 2, 1)],
        input_data={(0, 1): x, (1, 1): y},
        expected=expected,
    )


def _score_top_fanout(case: ScoreboardCase) -> ScoreboardResult:
    x = np.array([2, 4, 6, 8], dtype=np.int32)
    k0 = _compile_single_kernel(lambda t: t[0] + 1, [x])
    k1 = _compile_single_kernel(lambda t: t[0] * 2, [x])
    k2 = _compile_single_kernel(lambda t: t[0] * 3, [x])
    expected = (x + 1) * 3
    return _probe_manual_top(
        case,
        kernels=[k0, k1, k2],
        connections=[(0, 0, 1, 1), (0, 0, 2, 1)],
        input_data={(0, 1): x},
        expected=expected,
    )


def _score_top_float_chain(case: ScoreboardCase) -> ScoreboardResult:
    x = np.array([1.25, -0.5, 3.0, 2.5], dtype=np.float32)
    k0 = _compile_single_kernel(lambda t: t[0] + 1.0, [x])
    k1 = _compile_single_kernel(lambda t: t[0] * 2.0, [x])
    expected = (x + np.float32(1.0)) * np.float32(2.0)
    return _probe_manual_top(
        case,
        kernels=[k0, k1],
        connections=[(0, 0, 1, 1)],
        input_data={(0, 1): x},
        expected=expected,
        is_float=True,
    )


def _score_fp16_add(case: ScoreboardCase) -> ScoreboardResult:
    a = np.array([1.0, -2.0, 3.5, 0.25], dtype=np.float16)
    b = np.array([0.5, 4.0, -1.5, 2.0], dtype=np.float16)
    return _probe_bench(case, lambda t: t[0] + t[1], [a, b], exact=False)


def _score_bf16_relu(case: ScoreboardCase) -> ScoreboardResult:
    a = np.array([-1.0, 0.0, 3.0, -4.0], dtype=np.float32)
    return _probe_bench(case, lambda t: t[0].cast(dtypes.bfloat16).relu(), [a], exact=False)


def build_cases() -> list[ScoreboardCase]:
    return [
        ScoreboardCase("add_int32_scalar", 0, "kernel", "int32_add", "Scalar int32 add should stay green", lambda: _score_add_int32(_CASES["add_int32_scalar"])),
        ScoreboardCase("relu_int32_vector", 0, "kernel", "relu", "Vector int32 relu baseline", lambda: _score_relu_int32(_CASES["relu_int32_vector"])),
        ScoreboardCase("gemv_int8_small", 1, "kernel", "gemv_int8", "Small int8 GEMV baseline", lambda: _score_gemv_int8(_CASES["gemv_int8_small"])),
        ScoreboardCase("gemv_fp32_small", 1, "kernel", "gemv_fp32", "Small fp32 GEMV baseline", lambda: _score_gemv_fp32(_CASES["gemv_fp32_small"])),
        ScoreboardCase("mod_int32_vector", 1, "feature", "mod", "Vector modulo should remain correct", lambda: _score_mod_int32(_CASES["mod_int32_vector"])),
        ScoreboardCase("cmp_eq_int32_vector", 1, "feature", "cmp_eq", "Vector equality compare should remain correct", lambda: _score_cmp_eq_int32(_CASES["cmp_eq_int32_vector"])),
        ScoreboardCase("mlp_three_layer_chain", 2, "model", "three_layer_chain", "Three-layer chained MLP through sequential kernel simulation", lambda: _score_three_layer_chain(_CASES["mlp_three_layer_chain"])),
        ScoreboardCase("gemv_reduce_unroll_2x", 2, "optimizer", "reduce_unroll", "Reduce-axis unroll should still preserve correctness", lambda: _score_reduce_unroll(_CASES["gemv_reduce_unroll_2x"])),
        ScoreboardCase("top_adjacent_chain_int32", 2, "system", "topmodule_chain", "Adjacent kernel-to-kernel copy in TopModule", lambda: _score_top_adjacent_chain(_CASES["top_adjacent_chain_int32"])),
        ScoreboardCase("top_non_adjacent_dependency", 3, "system", "topmodule_non_adjacent_edge", "TopModule should support producer->consumer edges that skip intermediate kernels", lambda: _score_top_non_adjacent_dependency(_CASES["top_non_adjacent_dependency"])),
        ScoreboardCase("top_fanout_dependency", 3, "system", "topmodule_fanout", "TopModule should support one producer feeding two later kernels", lambda: _score_top_fanout(_CASES["top_fanout_dependency"])),
        ScoreboardCase("top_float_chain", 3, "system", "topmodule_float_io", "TopModule should load float inputs as bit patterns, not truncated ints", lambda: _score_top_float_chain(_CASES["top_float_chain"])),
        ScoreboardCase("fp16_add", 4, "dtype", "fp16", "Float16 arithmetic path", lambda: _score_fp16_add(_CASES["fp16_add"])),
        ScoreboardCase("bf16_relu", 4, "dtype", "bf16", "BFloat16 relu path", lambda: _score_bf16_relu(_CASES["bf16_relu"])),
    ]


_CASES = {case.name: case for case in build_cases()}


def iter_cases(*, max_tier: int | None = None, category: str | None = None, feature: str | None = None):
    for case in _CASES.values():
        if max_tier is not None and case.tier > max_tier:
            continue
        if category is not None and case.category != category:
            continue
        if feature is not None and case.feature != feature:
            continue
        yield case


def run_scoreboard(*, max_tier: int | None = None, category: str | None = None, feature: str | None = None) -> list[ScoreboardResult]:
    results = []
    for case in iter_cases(max_tier=max_tier, category=category, feature=feature):
        results.append(_run_case(case))
    return results


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 1e3:.2f}ms"


def _print_results(results: list[ScoreboardResult]) -> None:
    if not results:
        print("No scoreboard cases matched the current filters.")
        return

    header = f"{'status':<12} {'tier':<4} {'category':<10} {'feature':<28} {'cycles':>10} {'sim':>10} {'tg':>10}  name"
    print(header)
    print("-" * len(header))
    for r in results:
        cycles = "-" if r.cycles is None else f"{r.cycles}"
        print(
            f"{r.status:<12} {r.tier:<4} {r.category:<10} {r.feature:<28} "
            f"{cycles:>10} {_format_seconds(r.sim_wall_s):>10} {_format_seconds(r.tg_wall_s):>10}  {r.name}"
        )
        if r.detail:
            print(f"  {r.detail}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print()
    print(f"Summary: {summary}")


def _write_json(path: str | Path, results: list[ScoreboardResult]) -> None:
    payload = {
        "generated_at_s": time.time(),
        "results": [asdict(r) for r in results],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Project scoreboard for correctness, feature support, and performance")
    parser.add_argument("--max-tier", type=int, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--feature", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    results = run_scoreboard(max_tier=args.max_tier, category=args.category, feature=args.feature)
    _print_results(results)
    if args.json_out:
        _write_json(args.json_out, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
