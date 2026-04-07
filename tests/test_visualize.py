from tinygrad import Tensor, dtypes

from compiler.visualize import analyze_tensor


def test_analyze_tensor_reports_kernel_graph():
    x = Tensor.empty(1, 4, dtype=dtypes.int8)
    w1 = Tensor.empty(4, 3, dtype=dtypes.int8)
    b1 = Tensor.empty(1, 3, dtype=dtypes.int32)
    w2 = Tensor.empty(3, 2, dtype=dtypes.int8)
    b2 = Tensor.empty(1, 2, dtype=dtypes.int32)

    h = ((x @ w1).cast(dtypes.int32) + b1).relu()
    out = (h.cast(dtypes.int8) @ w2).cast(dtypes.int32) + b2

    view = analyze_tensor(out)
    text = view.to_text(include_uops=False, include_kernel_ir=False)

    assert len(view.kernel_views) == 2
    assert view.connections
    assert "PIPELINE" in text
    assert "K0.buf0 -> K1.buf1" in text
    assert "EXECUTION_PLAN" in text
    assert "K0_COPY_0" in text


def test_pipeline_dot_contains_nodes_and_edges():
    x = Tensor.empty(1, 4, dtype=dtypes.int8)
    w = Tensor.empty(4, 3, dtype=dtypes.int8)
    out = (x @ w).cast(dtypes.int32)

    view = analyze_tensor(out)
    dot = view.to_dot()

    assert "digraph tinygrad_pipeline" in dot
    assert "K0" in dot


def test_execution_dot_contains_execution_graph():
    x = Tensor.empty(1, 4, dtype=dtypes.int8)
    w1 = Tensor.empty(4, 3, dtype=dtypes.int8)
    b1 = Tensor.empty(1, 3, dtype=dtypes.int32)
    w2 = Tensor.empty(3, 2, dtype=dtypes.int8)
    b2 = Tensor.empty(1, 2, dtype=dtypes.int32)

    h = ((x @ w1).cast(dtypes.int32) + b1).relu()
    out = (h.cast(dtypes.int8) @ w2).cast(dtypes.int32) + b2
    view = analyze_tensor(out)

    dot = view.execution_dot()
    assert "digraph topmodule_execution" in dot
    assert "exec K0" in dot
