"""IR transform utilities."""

from dataclasses import dataclass

from tinygrad.uop.ops import AxisType

from .ir import DType, IRBufLoad, IRBufStore, IRConst, IRCounter, IROp, IRRegLoad, IRRegStore


@dataclass(frozen=True)
class LoopUnrollInfo:
    """Observability payload for LOOP unroll requests."""

    requested_unroll: int
    applied_unroll: int
    tail_length: int
    loop_bound: int
    iterations_per_cycle_est: float
    fallback_reason: str | None = None


def _substitute_lane_expr(val, *, lane: int, factor: int, target_depth: int, memo: dict):
    """Clone an IRValue and substitute loop counter with base*factor + lane."""
    if id(val) in memo:
        return memo[id(val)]

    if isinstance(val, IRConst):
        out = val
    elif isinstance(val, IRCounter):
        if val.depth != target_depth:
            out = val
        else:
            mul = IROp("mul", DType.INT32, (val, IRConst(factor, DType.INT32)))
            out = IROp("add", DType.INT32, (mul, IRConst(lane, DType.INT32)))
    elif isinstance(val, IRBufLoad):
        out = IRBufLoad(buf_idx=val.buf_idx, addr=_substitute_lane_expr(
            val.addr, lane=lane, factor=factor, target_depth=target_depth, memo=memo
        ))
    elif isinstance(val, IRRegLoad):
        out = val
    elif isinstance(val, IROp):
        out = IROp(
            val.op,
            val.dtype,
            tuple(
                _substitute_lane_expr(s, lane=lane, factor=factor, target_depth=target_depth, memo=memo)
                for s in val.srcs
            ),
        )
    else:
        out = val

    memo[id(val)] = out
    return out


def _clone_store_for_lane(store, *, lane: int, factor: int, depth: int):
    memo = {}
    if isinstance(store, IRBufStore):
        return IRBufStore(
            buf_idx=store.buf_idx,
            addr=_substitute_lane_expr(store.addr, lane=lane, factor=factor, target_depth=depth, memo=memo),
            value=_substitute_lane_expr(store.value, lane=lane, factor=factor, target_depth=depth, memo=memo),
            dtype=store.dtype,
        )
    if isinstance(store, IRRegStore):
        return IRRegStore(
            value=_substitute_lane_expr(store.value, lane=lane, factor=factor, target_depth=depth, memo=memo),
        )
    return store


def apply_loop_unroll(kernel_ir, unroll_loop: int) -> tuple:
    """Apply (or conservatively reject) LOOP-axis unrolling.

    This initial implementation adds guardrails/observability and only enables
    the transform when the requested factor is ``1``. Any request above ``1``
    currently falls back to scalar execution with an explicit reason, keeping
    correctness deterministic while the lane-replicated datapath/FSM is built.
    """
    requested = max(1, int(unroll_loop))
    loop_level = kernel_ir.loop_tree.body
    loop_bound = loop_level.bound if loop_level is not None else 0
    tail = loop_bound % requested if requested > 0 else 0

    if requested == 1:
        return kernel_ir, LoopUnrollInfo(
            requested_unroll=1,
            applied_unroll=1,
            tail_length=0,
            loop_bound=loop_bound,
            iterations_per_cycle_est=1.0,
            fallback_reason=None,
        )

    if loop_level is None:
        reason = "no loops in kernel"
    elif loop_level.axis_type != AxisType.LOOP:
        reason = "outermost axis is not AxisType.LOOP"
    elif loop_level.body is not None:
        reason = "nested loops not yet supported for unroll"
    elif any(isinstance(s, IRRegStore) for s in (loop_level.prologue + loop_level.epilogue)):
        reason = "register stores/reduce loops are not yet supported for unroll"
    elif tail != 0:
        reason = "tail handling for non-divisible loop bounds not yet enabled"
    else:
        # Supported v1: single LOOP-only level, divisible by U, memory stores only.
        new_bound = loop_bound // requested
        loop_level.bound = new_bound
        old_pro = list(loop_level.prologue)
        old_epi = list(loop_level.epilogue)
        loop_level.prologue = [
            _clone_store_for_lane(s, lane=lane, factor=requested, depth=loop_level.depth)
            for lane in range(requested)
            for s in old_pro
        ]
        loop_level.epilogue = [
            _clone_store_for_lane(s, lane=lane, factor=requested, depth=loop_level.depth)
            for lane in range(requested)
            for s in old_epi
        ]

        return kernel_ir, LoopUnrollInfo(
            requested_unroll=requested,
            applied_unroll=requested,
            tail_length=0,
            loop_bound=loop_bound,
            iterations_per_cycle_est=float(requested),
            fallback_reason=None,
        )

    return kernel_ir, LoopUnrollInfo(
        requested_unroll=requested,
        applied_unroll=1,
        tail_length=tail,
        loop_bound=loop_bound,
        iterations_per_cycle_est=1.0,
        fallback_reason=reason,
    )
