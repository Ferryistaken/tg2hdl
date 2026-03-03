"""IR transform utilities."""

from dataclasses import dataclass

from tinygrad.uop.ops import AxisType


@dataclass(frozen=True)
class LoopUnrollInfo:
    """Observability payload for LOOP unroll requests."""

    requested_unroll: int
    applied_unroll: int
    tail_length: int
    loop_bound: int
    iterations_per_cycle_est: float
    fallback_reason: str | None = None


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
    else:
        reason = "lane-replicated lowering/control not enabled yet"

    return kernel_ir, LoopUnrollInfo(
        requested_unroll=requested,
        applied_unroll=1,
        tail_length=tail,
        loop_bound=loop_bound,
        iterations_per_cycle_est=1.0,
        fallback_reason=reason,
    )

