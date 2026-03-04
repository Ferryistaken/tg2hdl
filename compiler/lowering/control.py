"""Control lowering: KernelIR + ArithResult → Amaranth FSM.

Builds the FSM that sequences stores (register updates and memory writes)
according to the loop structure in KernelIR.  Mirrors the behavior of the
old CompiledKernel._build_fsm() and _build_scalar_fsm() exactly, but
accepts typed IR nodes instead of raw UOps and an untyped sig dict.

State naming (unchanged from old design):
  IDLE         — wait for start
  L{d}_PRO     — non-innermost level prologue (e.g. acc reset)
  L{d}_BODY    — innermost level body (e.g. MAC)
  L{d}_EPI     — non-innermost level epilogue (e.g. output write)
  SCALAR       — single compute cycle for no-loop kernels
"""

from amaranth.hdl import Module, Signal, Const

from ..ir import IRBufStore, IRRegStore, LoopIR, KernelIR
from .arithmetic import ArithResult


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_control(
    m: Module,
    kernel: KernelIR,
    result: ArithResult,
    int_wports: dict,    # buf_idx → Amaranth write port
    start: Signal,
    done: Signal,
    busy: Signal,
) -> None:
    """Build the FSM into Module m.

    Parameters
    ----------
    m : Module
        The Amaranth module under construction.
    kernel : KernelIR
        Typed kernel IR (loop tree + stores).
    result : ArithResult
        Signals produced by ArithmeticLowering.run().
    int_wports : dict
        Maps buf_idx → Amaranth write port (from _create_memories).
    start, done, busy : Signal
        External control signals on the CompiledKernel.
    """
    # Flatten loop levels: list of (LoopIR, depth)
    levels = []
    level = kernel.loop_tree.body
    while level is not None:
        levels.append((level, level.depth))
        level = level.body

    # Scalar kernel (no loops)
    if not levels:
        root_stores = kernel.scalar_stores + [
            s for s in (kernel.loop_tree.prologue + kernel.loop_tree.epilogue)
            if isinstance(s, IRBufStore)
        ]
        _build_scalar_fsm(m, root_stores, result, int_wports, start, done)
        return

    # Determine first real state after IDLE
    outermost_level = levels[0][0]
    if outermost_level.body is None:
        first_state = "L0_BODY"
    else:
        first_state = "L0_PRO"

    with m.FSM(init="IDLE"):
        with m.State("IDLE"):
            m.d.sync += done.eq(0)
            with m.If(start):
                ctr = result.counter_sigs[levels[0][1]]
                m.d.sync += ctr.eq(0)
                m.next = first_state

        for i, (lvl, d) in enumerate(levels):
            is_innermost = (lvl.body is None)
            pro_stores = [s for s in lvl.prologue if isinstance(s, (IRBufStore, IRRegStore))]
            epi_stores = [s for s in lvl.epilogue if isinstance(s, (IRBufStore, IRRegStore))]
            ctr = result.counter_sigs[d]

            if is_innermost:
                with m.State(f"L{d}_BODY"):
                    _emit_stores(m, pro_stores, result, int_wports)
                    with m.If(ctr == lvl.bound - 1):
                        if d > 0:
                            m.next = f"L{d-1}_EPI"
                        else:
                            # Single-level loop: done when counter wraps
                            m.d.sync += done.eq(1)
                            m.next = "IDLE"
                    with m.Else():
                        m.d.sync += ctr.eq(ctr + 1)
                        m.next = f"L{d}_BODY"
            else:
                # Non-innermost: PRO (prologue) and EPI (epilogue) states
                child_level = levels[i + 1][0]
                child_d = levels[i + 1][1]
                if child_level.body is None:
                    child_first = f"L{child_d}_BODY"
                else:
                    child_first = f"L{child_d}_PRO"

                with m.State(f"L{d}_PRO"):
                    _emit_stores(m, pro_stores, result, int_wports)
                    child_ctr = result.counter_sigs[child_d]
                    m.d.sync += child_ctr.eq(0)
                    m.next = child_first

                with m.State(f"L{d}_EPI"):
                    _emit_stores(m, epi_stores, result, int_wports)
                    with m.If(ctr == lvl.bound - 1):
                        if d > 0:
                            m.next = f"L{d-1}_EPI"
                        else:
                            m.d.sync += done.eq(1)
                            m.next = "IDLE"
                    with m.Else():
                        m.d.sync += ctr.eq(ctr + 1)
                        m.next = f"L{d}_PRO"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_scalar_fsm(m, stores, result, int_wports, start, done):
    """FSM for scalar kernels (no loops). One compute cycle."""
    with m.FSM(init="IDLE"):
        with m.State("IDLE"):
            m.d.sync += done.eq(0)
            with m.If(start):
                m.next = "SCALAR"
        with m.State("SCALAR"):
            _emit_stores(m, stores, result, int_wports)
            m.d.sync += done.eq(1)
            m.next = "IDLE"


def _emit_stores(m, stores, result: ArithResult, int_wports: dict):
    """Emit side effects (register writes + memory writes) for a list of stores.

    Called within an FSM state context (inside m.State(...)).
    """
    used_wports = {}
    for store in stores:
        if isinstance(store, IRRegStore):
            # Accumulator register update
            value_sig = result.signals.get(id(store.value))
            if value_sig is None:
                raise RuntimeError(
                    f"_emit_stores: IRRegStore has no emitted signal for its value "
                    f"{store.value!r}. ArithmeticLowering did not emit this IRValue."
                )
            if result.acc is None:
                raise RuntimeError(
                    "_emit_stores: IRRegStore encountered but ArithResult.acc is None. "
                    "The kernel has no accumulator register but the IR contains a "
                    "register store — the UOp graph is inconsistent."
                )
            m.d.sync += result.acc.eq(value_sig)

        elif isinstance(store, IRBufStore):
            # Memory write — override default (external-load) write port wiring
            addr_sig = result.signals.get(id(store.addr))
            if addr_sig is None:
                raise RuntimeError(
                    f"_emit_stores: IRBufStore(buf_idx={store.buf_idx}) has no emitted "
                    f"signal for its address expression {store.addr!r}. "
                    "ArithmeticLowering did not emit this IRValue."
                )
            value_sig = result.signals.get(id(store.value))
            if value_sig is None:
                raise RuntimeError(
                    f"_emit_stores: IRBufStore(buf_idx={store.buf_idx}) has no emitted "
                    f"signal for its value expression {store.value!r}. "
                    "ArithmeticLowering did not emit this IRValue."
                )
            wps = int_wports.get(store.buf_idx)
            if not wps:
                raise RuntimeError(
                    f"_emit_stores: IRBufStore targets buf_idx={store.buf_idx} "
                    "but no write port exists for that buffer index. "
                    "Check that _create_memories() created a write port for every buffer."
                )
            if len(wps) == 1:
                wp = wps[0]
            else:
                wp_idx = used_wports.get(store.buf_idx, 0)
                if wp_idx >= len(wps):
                    raise RuntimeError(
                        f"_emit_stores: IRBufStore targets buf_idx={store.buf_idx} requires "
                        f"{wp_idx + 1} concurrent write ports but only {len(wps)} exist."
                    )
                wp = wps[wp_idx]
                used_wports[store.buf_idx] = wp_idx + 1
            m.d.comb += [
                wp.addr.eq(addr_sig),
                wp.data.eq(value_sig),
                wp.en.eq(1),
            ]
