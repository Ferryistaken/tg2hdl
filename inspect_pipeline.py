from __future__ import annotations

import argparse
from pathlib import Path

from tinygrad import Tensor, dtypes

from compiler import HDLRenderer, compile_kernel
from compiler.backend import _get_uops
from compiler.visualize import analyze_manual_kernels, analyze_tensor


def _build_example(name: str):
    if name == "gemv_int8":
        x = Tensor.empty(1, 4, dtype=dtypes.int8)
        w = Tensor.empty(4, 3, dtype=dtypes.int8)
        return (x @ w).cast(dtypes.int32)
    if name == "mlp_2layer":
        x = Tensor.empty(1, 4, dtype=dtypes.int8)
        w1 = Tensor.empty(4, 3, dtype=dtypes.int8)
        b1 = Tensor.empty(1, 3, dtype=dtypes.int32)
        w2 = Tensor.empty(3, 2, dtype=dtypes.int8)
        b2 = Tensor.empty(1, 2, dtype=dtypes.int32)
        h = ((x @ w1).cast(dtypes.int32) + b1).relu()
        return (h.cast(dtypes.int8) @ w2).cast(dtypes.int32) + b2
    if name == "mlp_3layer":
        x = Tensor.empty(1, 6, dtype=dtypes.int8)
        w1 = Tensor.empty(6, 5, dtype=dtypes.int8)
        b1 = Tensor.empty(1, 5, dtype=dtypes.int32)
        w2 = Tensor.empty(5, 4, dtype=dtypes.int8)
        b2 = Tensor.empty(1, 4, dtype=dtypes.int32)
        w3 = Tensor.empty(4, 3, dtype=dtypes.int8)
        b3 = Tensor.empty(1, 3, dtype=dtypes.int32)
        h1 = ((x @ w1).cast(dtypes.int32) + b1).relu()
        h2 = ((h1.cast(dtypes.int8) @ w2).cast(dtypes.int32) + b2).relu()
        return (h2.cast(dtypes.int8) @ w3).cast(dtypes.int32) + b3
    if name == "residual_add":
        x = Tensor.empty(1, 4, dtype=dtypes.int8)
        w = Tensor.empty(4, 4, dtype=dtypes.int8)
        b = Tensor.empty(1, 4, dtype=dtypes.int32)
        fx = ((x @ w).cast(dtypes.int32) + b).relu()
        return fx + x.cast(dtypes.int32)
    if name == "fanout_merge":
        x = Tensor.empty(1, 4, dtype=dtypes.int8)
        w_shared = Tensor.empty(4, 3, dtype=dtypes.int8)
        b_shared = Tensor.empty(1, 3, dtype=dtypes.int32)
        w_left = Tensor.empty(3, 2, dtype=dtypes.int8)
        b_left = Tensor.empty(1, 2, dtype=dtypes.int32)
        w_right = Tensor.empty(3, 2, dtype=dtypes.int8)
        b_right = Tensor.empty(1, 2, dtype=dtypes.int32)
        h = ((x @ w_shared).cast(dtypes.int32) + b_shared).relu()
        left = (h.cast(dtypes.int8) @ w_left).cast(dtypes.int32) + b_left
        right = (h.cast(dtypes.int8) @ w_right).cast(dtypes.int32) + b_right
        return left + right
    raise ValueError(f"Unknown example {name!r}")


def _compile_single_kernel(build_fn, arrays):
    syms = []
    for a in arrays:
        if isinstance(a, Tensor):
            syms.append(Tensor.empty(a.shape, dtype=a.dtype))
        else:
            syms.append(Tensor.empty(a.shape, dtype=Tensor(a).dtype))
    expr = build_fn(syms)
    schedule = expr.schedule()
    compute_items = [si for si in schedule if si.ast.op.name == "SINK"]
    if len(compute_items) != 1:
        raise RuntimeError(f"Expected exactly one compute kernel, got {len(compute_items)}")
    uops = _get_uops(compute_items[0].ast, HDLRenderer())
    return compile_kernel(uops)


def _build_manual_example(name: str):
    x = Tensor.empty(4, dtype=dtypes.int32)
    y = Tensor.empty(4, dtype=dtypes.int32)

    if name == "manual_non_adjacent":
        k0 = _compile_single_kernel(lambda t: t[0] + 1, [x])
        k1 = _compile_single_kernel(lambda t: t[0] + 2, [y])
        k2 = _compile_single_kernel(lambda t: t[0] * 3, [x])
        return analyze_manual_kernels(
            [k0, k1, k2],
            connections=[(0, 0, 2, 1)],
        )

    if name == "manual_fanout":
        k0 = _compile_single_kernel(lambda t: t[0] + 1, [x])
        k1 = _compile_single_kernel(lambda t: t[0] * 2, [x])
        k2 = _compile_single_kernel(lambda t: t[0] * 3, [x])
        return analyze_manual_kernels(
            [k0, k1, k2],
            connections=[(0, 0, 1, 1), (0, 0, 2, 1)],
        )

    raise ValueError(f"Unknown manual example {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize how tinygrad is interpreted by the HDL pipeline")
    parser.add_argument(
        "--example",
        choices=[
            "gemv_int8",
            "mlp_2layer",
            "mlp_3layer",
            "residual_add",
            "fanout_merge",
            "manual_non_adjacent",
            "manual_fanout",
        ],
        default="mlp_2layer",
    )
    parser.add_argument("--dot-out", type=str, default=None)
    parser.add_argument("--execution-dot-out", type=str, default=None)
    parser.add_argument("--no-uops", action="store_true")
    parser.add_argument("--no-kernel-ir", action="store_true")
    parser.add_argument("--full-width-uops", action="store_true")
    args = parser.parse_args()

    if args.example.startswith("manual_"):
        view = _build_manual_example(args.example)
    else:
        tensor = _build_example(args.example)
        view = analyze_tensor(tensor)
    print(view.to_text(
        include_uops=not args.no_uops,
        include_kernel_ir=not args.no_kernel_ir,
        full_width_uops=args.full_width_uops,
    ))

    if args.dot_out:
        Path(args.dot_out).write_text(view.to_dot() + "\n")
        print()
        print(f"Wrote DOT graph to {args.dot_out}")
    if args.execution_dot_out:
        Path(args.execution_dot_out).write_text(view.execution_dot() + "\n")
        print(f"Wrote execution DOT graph to {args.execution_dot_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
