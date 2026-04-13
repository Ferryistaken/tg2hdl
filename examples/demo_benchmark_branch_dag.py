import os
import sys
from pathlib import Path

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tg2hdl


os.environ.setdefault("NOOPT", "1")
os.environ.setdefault("DEBUG", "0")


def build_graph():
    rng = np.random.RandomState(2)
    x = Tensor(rng.randint(-4, 4, (1, 8)).astype(np.int8))
    w1 = Tensor(rng.randint(-3, 3, (8, 8)).astype(np.int8))
    w2 = Tensor(rng.randint(-3, 3, (8, 4)).astype(np.int8))
    w3 = Tensor(rng.randint(-3, 3, (8, 5)).astype(np.int8))
    b1 = Tensor(rng.randint(-6, 6, (1, 8)).astype(np.int32))
    b2 = Tensor(rng.randint(-6, 6, (1, 4)).astype(np.int32))
    b3 = Tensor(rng.randint(-6, 6, (1, 5)).astype(np.int32))

    h = ((x @ w1).cast(dtypes.int32) + b1).relu()
    y1 = ((h.cast(dtypes.int8) @ w2).cast(dtypes.int32) + b2).relu()
    y2 = ((h.cast(dtypes.int8) @ w3).cast(dtypes.int32) + b3).relu()
    return y1, y2


def main() -> None:
    y1_for_count, y2_for_count = build_graph()
    schedule = y1_for_count.schedule(y2_for_count)
    kernel_count = len([si for si in schedule if si.ast.op == Ops.SINK])

    y1, y2 = build_graph()
    artifact = tg2hdl.benchmark(
        y2,
        schedule_outputs=[y1, y2],
        out_dir="tmp/demo_report_branch_dag",
    )

    print("graph_shape: tinygrad fanout")
    print(f"kernels: {kernel_count}")
    print(f"report: {artifact.report_path}")
    print(f"correct: {artifact.correctness}")
    print(f"tinygrad_device: {artifact.tinygrad_device}")
    print(f"tinygrad_wall_s: {artifact.tinygrad_wall_s:.6f}")
    print(f"tg2hdl_wall_s: {artifact.tg2hdl_wall_s:.6f}")
    print(f"tg2hdl_cycles: {artifact.tg2hdl_cycles}")
    print(f"tg2hdl_total_cycles: {artifact.tg2hdl_total_cycles}")
    print(f"fpga_wall_s: {artifact.fpga_wall_s}")
    print(f"report_exists: {Path(artifact.report_path).exists()}")


if __name__ == "__main__":
    main()
