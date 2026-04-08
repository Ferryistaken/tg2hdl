import os
import sys
import time
from pathlib import Path

import numpy as np
from tinygrad import Tensor
from tinygrad.uop.ops import Ops
from tinygrad.viz.serve import uop_to_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tg2hdl
from compiler import HDLRenderer, compile_kernel
from compiler.backend import _get_uops


os.environ.setdefault("NOOPT", "1")
os.environ.setdefault("DEBUG", "0")


def compile_single_kernel(build_fn, arrays):
    syms = [Tensor.empty(a.shape, dtype=Tensor(a).dtype) for a in arrays]
    expr = build_fn(syms)
    schedule = expr.schedule()
    compute_items = [si for si in schedule if si.ast.op == Ops.SINK]
    assert len(compute_items) == 1
    item = compute_items[0]
    uops = _get_uops(item.ast, HDLRenderer())
    return compile_kernel(uops), uop_to_json(item.ast), tuple(str(m.name if hasattr(m, "name") else m) for m in item.metadata)


def main() -> None:
    x = np.array([2, 4, 6, 8], dtype=np.int32)

    k0, g0, m0 = compile_single_kernel(lambda t: t[0] + 1, [x])
    k1, g1, m1 = compile_single_kernel(lambda t: t[0] * 2, [x])
    k2, g2, m2 = compile_single_kernel(lambda t: t[0] * 3, [x])
    k3, g3, m3 = compile_single_kernel(lambda t: t[0] + t[1], [x, x])

    t0 = time.perf_counter()
    ref = ((((Tensor(x) + 1) * 2) + ((Tensor(x) + 1) * 3)).numpy()).astype(np.int32).reshape(-1)
    ref_wall = time.perf_counter() - t0

    artifact = tg2hdl.benchmark_manual(
        kernels=[k0, k1, k2, k3],
        connections=[(0, 0, 1, 1), (0, 0, 2, 1), (1, 0, 3, 1), (2, 0, 3, 2)],
        input_data={(0, 1): x},
        reference_output=ref,
        reference_wall_s=ref_wall,
        tinygrad_device="CPU",
        tinygrad_graphs=[g0, g1, g2, g3],
        metadata=[m0, m1, m2, m3],
        out_dir="tmp/demo_report_branch_dag",
    )

    print("graph_shape: branch-merge")
    print("kernels: 4")
    print(f"report: {artifact.report_path}")
    print(f"correct: {artifact.correctness}")
    print(f"tinygrad_device: {artifact.tinygrad_device}")
    print(f"tinygrad_wall_s: {artifact.tinygrad_wall_s:.6f}")
    print(f"tg2hdl_wall_s: {artifact.tg2hdl_wall_s:.6f}")
    print(f"tg2hdl_cycles: {artifact.tg2hdl_cycles}")
    print(f"report_exists: {Path(artifact.report_path).exists()}")


if __name__ == "__main__":
    main()
