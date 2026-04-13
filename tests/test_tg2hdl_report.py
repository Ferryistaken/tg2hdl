import os

os.environ["NOOPT"] = "1"
os.environ["DEBUG"] = "0"

import numpy as np
from tinygrad import Tensor, dtypes

import tg2hdl


def test_benchmark_generates_html_report(tmp_path):
    x = Tensor(np.array([[1, 2, 3, 4]], dtype=np.int8))
    w = Tensor(np.array([[1, 0, -1], [0, 1, 0], [1, 1, 1], [-1, 0, 1]], dtype=np.int8))
    b = Tensor(np.array([[1, -2, 3]], dtype=np.int32))
    out = ((x @ w).cast(dtypes.int32) + b).relu()

    artifact = tg2hdl.benchmark(out, out_dir=str(tmp_path / "report"))

    assert artifact.correctness
    assert (tmp_path / "report" / "index.html").exists()
    html = (tmp_path / "report" / "index.html").read_text()
    assert "tg2hdl Report" in html
    assert "Kernel DAG" in html
