import os
import sys
from pathlib import Path

import numpy as np
from tinygrad import Tensor, dtypes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tg2hdl


os.environ.setdefault("NOOPT", "1")
os.environ.setdefault("DEBUG", "0")


def main() -> None:
    x = Tensor(np.array([[1, 2, 3, 4]], dtype=np.int8))
    w1 = Tensor(np.array([
        [1, 0, -1],
        [0, 1, 0],
        [1, 1, 1],
        [-1, 0, 1],
    ], dtype=np.int8))
    b1 = Tensor(np.array([[1, -2, 3]], dtype=np.int32))

    out = ((x @ w1).cast(dtypes.int32) + b1).relu()
    artifact = tg2hdl.benchmark(out, out_dir="tmp/demo_report")

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
