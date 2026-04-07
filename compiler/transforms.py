"""KernelIR → KernelIR transforms.

Pure IR transforms applied between uop_to_ir() and Amaranth lowering.
No Amaranth imports — this module is purely structural.
"""

from .ir import (
    DType,
    IRConst, IRCounter, IRBufLoad, IRRegLoad, IROp,
    IRBufStore, IRRegStore,
    LoopIR, KernelIR,
)


# ---------------------------------------------------------------------------
# IRValue substitution helper
# ---------------------------------------------------------------------------

def _substitute(val, target_depth, expr_fn, memo=None):
    """Walk an IRValue DAG and replace IRCounter nodes at target_depth.

    Parameters
    ----------
    val : IRValue
        The root of the DAG to transform.
    target_depth : int
        The loop depth whose counter should be replaced.
    expr_fn : callable(IRCounter) -> IRValue
        Called when an IRCounter at target_depth is found.
        Returns the replacement IRValue.
    memo : dict or None
        Memoization cache keyed by id(val). Created if None.

    Returns
    -------
    IRValue
        The transformed DAG (shares unmodified subtrees).
    """
    if memo is None:
        memo = {}
    key = id(val)
    if key in memo:
        return memo[key]

    if isinstance(val, IRCounter):
        if val.depth == target_depth:
            result = expr_fn(val)
        else:
            result = val
    elif isinstance(val, (IRConst, IRRegLoad)):
        result = val
    elif isinstance(val, IRBufLoad):
        new_addr = _substitute(val.addr, target_depth, expr_fn, memo)
        if new_addr is val.addr:
            result = val
        else:
            result = IRBufLoad(buf_idx=val.buf_idx, addr=new_addr)
    elif isinstance(val, IROp):
        new_srcs = tuple(
            _substitute(s, target_depth, expr_fn, memo) for s in val.srcs
        )
        if all(ns is os for ns, os in zip(new_srcs, val.srcs)):
            result = val
        else:
            result = IROp(op=val.op, dtype=val.dtype, srcs=new_srcs)
    else:
        result = val

    memo[key] = result
    return result


def _substitute_store(store, target_depth, expr_fn, memo):
    """Substitute counters in a store node."""
    if isinstance(store, IRRegStore):
        new_val = _substitute(store.value, target_depth, expr_fn, memo)
        return IRRegStore(value=new_val)
    elif isinstance(store, IRBufStore):
        new_addr = _substitute(store.addr, target_depth, expr_fn, memo)
        new_val = _substitute(store.value, target_depth, expr_fn, memo)
        return IRBufStore(
            buf_idx=store.buf_idx,
            addr=new_addr,
            value=new_val,
            dtype=store.dtype,
        )
    return store


# ---------------------------------------------------------------------------
# unroll_loop
# ---------------------------------------------------------------------------

def unroll_loop(kernel_ir, depth, factor):
    """Unroll a LOOP axis by the given factor.

    Parameters
    ----------
    kernel_ir : KernelIR
        Input kernel IR.
    depth : int
        Which loop level to unroll (0 = outermost).
    factor : int
        Unroll factor N. Must divide the loop bound.

    Returns
    -------
    KernelIR
        New KernelIR with the target loop unrolled.

    Raises
    ------
    ValueError
        If depth not found, axis is not LOOP, or factor doesn't divide bound.
    """
    if factor <= 1:
        return kernel_ir

    # Find the target LoopIR node and its parent
    target, parent = _find_loop(kernel_ir.loop_tree, depth)
    if target is None:
        raise ValueError(f"No loop at depth {depth}")

    from tinygrad.uop.ops import AxisType
    if target.axis_type != AxisType.LOOP:
        raise ValueError(
            f"Loop at depth {depth} has axis_type={target.axis_type}, "
            "only LOOP axes can be unrolled (not REDUCE)."
        )

    if target.bound % factor != 0:
        raise ValueError(
            f"Factor {factor} does not divide loop bound {target.bound}."
        )

    new_bound = target.bound // factor

    all_stores = target.prologue + target.epilogue
    if target.body is not None:
        raise NotImplementedError(
            "Unrolling a loop that contains nested loops is not yet supported. "
            "Unroll the innermost loop first."
        )

    if new_bound == 1:
        # Full unroll — eliminate the loop, substitute counter with CONST(k)
        unrolled_stores = []
        for k in range(factor):
            memo = {}
            def make_const(counter, _k=k):
                return IRConst(_k, DType.INT32)
            for store in all_stores:
                unrolled_stores.append(
                    _substitute_store(store, depth, make_const, memo)
                )

        new_root, parent_copy = _copy_spine(kernel_ir.loop_tree, depth)
        if parent_copy is not None:
            parent_copy.body = None
            parent_copy.epilogue = parent_copy.epilogue + unrolled_stores
        else:
            new_root.body = None
            new_root.epilogue = new_root.epilogue + unrolled_stores

        scalar_stores = [
            s for s in (new_root.prologue + new_root.epilogue)
            if isinstance(s, IRBufStore)
        ]
        return KernelIR(
            buffers=kernel_ir.buffers,
            acc_dtype=kernel_ir.acc_dtype,
            loop_tree=new_root,
            scalar_stores=scalar_stores,
        )

    # Partial unroll — substitute counter j with j*N + k for each lane k
    unrolled_stores = []
    for k in range(factor):
        memo = {}

        def make_expr(counter, _k=k):
            new_counter = IRCounter(bound=new_bound, depth=counter.depth)
            scaled = IROp("mul", DType.INT32, (
                new_counter,
                IRConst(factor, DType.INT32),
            ))
            if _k == 0:
                return scaled
            return IROp("add", DType.INT32, (
                scaled,
                IRConst(_k, DType.INT32),
            ))

        for store in all_stores:
            unrolled_stores.append(_substitute_store(store, depth, make_expr, memo))

    new_target = LoopIR(
        axis_type=target.axis_type,
        bound=new_bound,
        depth=target.depth,
        prologue=unrolled_stores,
        body=None,
        epilogue=[],
    )

    new_root, parent_copy = _copy_spine(kernel_ir.loop_tree, depth)
    if parent_copy is not None:
        parent_copy.body = new_target
    else:
        new_root.body = new_target

    return KernelIR(
        buffers=kernel_ir.buffers,
        acc_dtype=kernel_ir.acc_dtype,
        loop_tree=new_root,
        scalar_stores=kernel_ir.scalar_stores,
    )


# ---------------------------------------------------------------------------
# unroll_reduce
# ---------------------------------------------------------------------------

def unroll_reduce(kernel_ir, depth, factor):
    """Unroll a REDUCE axis by the given factor.

    Widens the per-iteration accumulator expression so N independent
    contributions are computed in parallel and summed before adding to acc.

    Parameters
    ----------
    kernel_ir : KernelIR
        Input kernel IR.
    depth : int
        Loop depth of the REDUCE axis to unroll.
    factor : int
        Unroll factor N. Must divide the REDUCE bound.

    Returns
    -------
    KernelIR
        New KernelIR with the target REDUCE unrolled.
    """
    if factor <= 1:
        return kernel_ir

    target, parent = _find_loop(kernel_ir.loop_tree, depth)
    if target is None:
        raise ValueError(f"No loop at depth {depth}")

    from tinygrad.uop.ops import AxisType
    if target.axis_type != AxisType.REDUCE:
        raise ValueError(
            f"Loop at depth {depth} has axis_type={target.axis_type}, "
            "only REDUCE axes can be unrolled with unroll_reduce."
        )

    if target.body is not None:
        raise NotImplementedError(
            "Unrolling a REDUCE that contains nested loops is not supported."
        )

    if target.bound % factor != 0:
        raise ValueError(
            f"Factor {factor} does not divide REDUCE bound {target.bound}."
        )

    # Find the accumulator store and extract the contribution expression
    acc_store, acc_regload, contribution, add_dtype = _detect_acc_pattern(
        target.prologue
    )
    other_stores = [s for s in target.prologue if s is not acc_store]

    new_bound = target.bound // factor

    # Create N lane copies of the contribution with substituted counters
    lane_contribs = []
    for k in range(factor):
        memo = {}

        def make_expr(counter, _k=k):
            new_counter = IRCounter(bound=new_bound, depth=counter.depth)
            scaled = IROp("mul", DType.INT32, (
                new_counter,
                IRConst(factor, DType.INT32),
            ))
            if _k == 0:
                return scaled
            return IROp("add", DType.INT32, (
                scaled,
                IRConst(_k, DType.INT32),
            ))

        lane_contribs.append(_substitute(contribution, depth, make_expr, memo))

    # Build reduction tree and new accumulator store
    partial_sum = _build_reduction_tree(lane_contribs, add_dtype)
    new_value = IROp("add", add_dtype, (acc_regload, partial_sum))
    new_store = IRRegStore(value=new_value)

    # Substitute counters in other (non-accumulator) stores too
    new_other = []
    for k in range(factor):
        memo = {}

        def make_expr2(counter, _k=k):
            new_counter = IRCounter(bound=new_bound, depth=counter.depth)
            scaled = IROp("mul", DType.INT32, (
                new_counter,
                IRConst(factor, DType.INT32),
            ))
            if _k == 0:
                return scaled
            return IROp("add", DType.INT32, (
                scaled,
                IRConst(_k, DType.INT32),
            ))

        for store in other_stores:
            new_other.append(_substitute_store(store, depth, make_expr2, memo))

    new_target = LoopIR(
        axis_type=target.axis_type,
        bound=new_bound,
        depth=target.depth,
        prologue=[new_store] + new_other,
        body=None,
        epilogue=list(target.epilogue),
    )

    new_root, parent_copy = _copy_spine(kernel_ir.loop_tree, depth)
    if parent_copy is not None:
        parent_copy.body = new_target
    else:
        new_root.body = new_target

    return KernelIR(
        buffers=kernel_ir.buffers,
        acc_dtype=kernel_ir.acc_dtype,
        loop_tree=new_root,
        scalar_stores=kernel_ir.scalar_stores,
    )


def _detect_acc_pattern(stores):
    """Find the accumulator pattern in REDUCE body stores.

    Looks for: IRRegStore(IROp("add", (IRRegLoad, contribution)))
    Returns (store, regload, contribution, dtype).
    """
    for store in stores:
        if not isinstance(store, IRRegStore):
            continue
        val = store.value
        if not isinstance(val, IROp) or val.op != "add":
            continue
        lhs, rhs = val.srcs
        if isinstance(lhs, IRRegLoad) and not _contains_regload(rhs):
            return store, lhs, rhs, val.dtype
        if isinstance(rhs, IRRegLoad) and not _contains_regload(lhs):
            return store, rhs, lhs, val.dtype
    raise ValueError(
        "No accumulator pattern found in REDUCE body. "
        "Expected IRRegStore(IROp('add', (IRRegLoad, contribution)))."
    )


def _contains_regload(val):
    """Check if an IRValue DAG contains any IRRegLoad nodes."""
    if isinstance(val, IRRegLoad):
        return True
    if isinstance(val, IROp):
        return any(_contains_regload(s) for s in val.srcs)
    if isinstance(val, IRBufLoad):
        return _contains_regload(val.addr)
    return False


def _build_reduction_tree(contribs, dtype):
    """Build a right-associative add tree from N contributions."""
    assert len(contribs) >= 1
    if len(contribs) == 1:
        return contribs[0]
    return IROp("add", dtype, (
        contribs[0],
        _build_reduction_tree(contribs[1:], dtype),
    ))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_loop(root, target_depth):
    """Find the LoopIR at target_depth and its parent. Returns (target, parent)."""
    parent = root
    level = root.body
    while level is not None:
        if level.depth == target_depth:
            return level, parent
        parent = level
        level = level.body
    return None, None


def _copy_spine(root, target_depth):
    """Shallow-copy the LoopIR spine from root down to (and including) the parent of target_depth.

    IRValue DAGs in prologue/epilogue are frozen and shared by reference.
    Returns (new_root, parent_of_target) where parent_of_target is the copied
    node whose .body points at the target depth (or None if target is root's direct child).
    """
    new_root = LoopIR(
        axis_type=root.axis_type, bound=root.bound, depth=root.depth,
        prologue=list(root.prologue), body=root.body,
        epilogue=list(root.epilogue),
    )
    prev = new_root
    level = root.body
    while level is not None:
        if level.depth == target_depth:
            return new_root, prev
        copy = LoopIR(
            axis_type=level.axis_type, bound=level.bound, depth=level.depth,
            prologue=list(level.prologue), body=level.body,
            epilogue=list(level.epilogue),
        )
        prev.body = copy
        prev = copy
        level = level.body
    return new_root, None
