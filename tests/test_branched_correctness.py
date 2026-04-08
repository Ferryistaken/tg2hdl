import os

os.environ["NOOPT"] = "1"
os.environ["DEBUG"] = "0"

import numpy as np
from tinygrad import Tensor

from compiler import HDLRenderer, compile_kernel
from compiler.backend import _get_uops
from compiler.top_module import TopModule, simulate_top


def _compile_single_kernel(build_fn, arrays):
    syms = [Tensor.empty(a.shape, dtype=Tensor(a).dtype) for a in arrays]
    expr = build_fn(syms)
    schedule = expr.schedule()
    compute_items = [si for si in schedule if si.ast.op.name == "SINK"]
    assert len(compute_items) == 1
    uops = _get_uops(compute_items[0].ast, HDLRenderer())
    return compile_kernel(uops)


def test_manual_branch_merge_matches_tinygrad_cpu_reference():
    x = np.array([2, 4, 6, 8], dtype=np.int32)

    k0 = _compile_single_kernel(lambda t: t[0] + 1, [x])
    k1 = _compile_single_kernel(lambda t: t[0] * 2, [x])
    k2 = _compile_single_kernel(lambda t: t[0] * 3, [x])
    k3 = _compile_single_kernel(lambda t: t[0] + t[1], [x, x])

    top = TopModule(
        [k0, k1, k2, k3],
        connections=[(0, 0, 1, 1), (0, 0, 2, 1), (1, 0, 3, 1), (2, 0, 3, 2)],
        buf_depths={
            (0, 0): 4, (0, 1): 4,
            (1, 0): 4, (1, 1): 4,
            (2, 0): 4, (2, 1): 4,
            (3, 0): 4, (3, 1): 4, (3, 2): 4,
        },
    )

    sim_out, _, _ = simulate_top(top, {(0, 1): x})
    ref = ((((Tensor(x) + 1) * 2) + ((Tensor(x) + 1) * 3)).numpy()).astype(np.int32).reshape(-1)

    np.testing.assert_array_equal(sim_out, ref)
