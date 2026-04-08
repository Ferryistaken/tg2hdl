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


def build_model():
    rng = np.random.RandomState(0)

    x = Tensor(rng.randint(-4, 4, (1, 4)).astype(np.int8))
    w1 = Tensor(rng.randint(-3, 3, (4, 3)).astype(np.int8))
    b1 = Tensor(rng.randint(-6, 6, (1, 3)).astype(np.int32))
    w2 = Tensor(rng.randint(-3, 3, (3, 3)).astype(np.int8))
    b2 = Tensor(rng.randint(-6, 6, (1, 3)).astype(np.int32))
    w3 = Tensor(rng.randint(-3, 3, (3, 2)).astype(np.int8))
    b3 = Tensor(rng.randint(-6, 6, (1, 2)).astype(np.int32))

    h1 = ((x @ w1).cast(dtypes.int32) + b1).relu()
    h2 = ((h1.cast(dtypes.int8) @ w2).cast(dtypes.int32) + b2).relu()
    return (h2.cast(dtypes.int8) @ w3).cast(dtypes.int32) + b3


def main() -> None:
    out_for_count = build_model()
    schedule = out_for_count.schedule()
    kernel_count = len([si for si in schedule if si.ast.op == Ops.SINK])

    out = build_model()
    artifact = tg2hdl.benchmark(out, out_dir="tmp/demo_report_mlp3")

    print(f"kernels: {kernel_count}")
    print(f"report: {artifact.report_path}")
    print(f"correct: {artifact.correctness}")
    print(f"tinygrad_device: {artifact.tinygrad_device}")
    print(f"tinygrad_wall_s: {artifact.tinygrad_wall_s:.6f}")
    print(f"tg2hdl_wall_s: {artifact.tg2hdl_wall_s:.6f}")
    print(f"tg2hdl_cycles: {artifact.tg2hdl_cycles}")
    print(f"report_exists: {Path(artifact.report_path).exists()}")


if __name__ == "__main__":
    main()
