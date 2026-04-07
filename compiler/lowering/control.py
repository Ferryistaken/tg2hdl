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
        # De-duplicate: scalar_stores already extracted from root prologue+epilogue
        seen = set()
        root_stores = []
        for s in kernel.scalar_stores:
            if id(s) not in seen:
                seen.add(id(s))
                root_stores.append(s)
        for s in (kernel.loop_tree.prologue + kernel.loop_tree.epilogue):
            if isinstance(s, IRBufStore) and id(s) not in seen:
                seen.add(id(s))
                root_stores.append(s)
        _build_scalar_fsm(m, root_stores, result, int_wports, start, done)
        return

    # Pre-compute wave groups for innermost levels
    level_waves = {}  # depth → list of wave groups
    for lvl, d in levels:
        if lvl.body is None:  # innermost
            stores = [s for s in lvl.prologue if isinstance(s, (IRBufStore, IRRegStore))]
            level_waves[d] = _group_stores_by_wave(stores)

    def body_first_state(d):
        """Return the first body state name for level d."""
        waves = level_waves.get(d, [[]])
        return f"L{d}_BODY_0" if len(waves) > 1 else f"L{d}_BODY"

    # Determine first real state after IDLE
    outermost_level = levels[0][0]
    if outermost_level.body is None:
        first_state = body_first_state(0)
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
                waves = level_waves[d]

                if len(waves) <= 1:
                    with m.State(f"L{d}_BODY"):
                        _emit_stores(m, pro_stores, result, int_wports)
                        _emit_counter_check(m, ctr, lvl.bound, d, done, f"L{d}_BODY")
                else:
                    for wi, wave in enumerate(waves):
                        with m.State(f"L{d}_BODY_{wi}"):
                            _emit_stores(m, wave, result, int_wports)
                            if wi < len(waves) - 1:
                                m.next = f"L{d}_BODY_{wi + 1}"
                            else:
                                _emit_counter_check(m, ctr, lvl.bound, d, done, f"L{d}_BODY_0")
            else:
                # Non-innermost: PRO (prologue) and EPI (epilogue) states
                child_level = levels[i + 1][0]
                child_d = levels[i + 1][1]
                if child_level.body is None:
                    child_first = body_first_state(child_d)
                else:
                    child_first = f"L{child_d}_PRO"

                with m.State(f"L{d}_PRO"):
                    _emit_stores(m, pro_stores, result, int_wports)
                    child_ctr = result.counter_sigs[child_d]
                    m.d.sync += child_ctr.eq(0)
                    m.next = child_first

                with m.State(f"L{d}_EPI"):
                    _emit_stores(m, epi_stores, result, int_wports)
                    _emit_counter_check(m, ctr, lvl.bound, d, done, f"L{d}_PRO")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _emit_counter_check(m, ctr, bound, d, done, loop_back_state):
    """Emit the counter check + transition logic (shared by all body states)."""
    with m.If(ctr == bound - 1):
        if d > 0:
            m.next = f"L{d-1}_EPI"
        else:
            m.d.sync += done.eq(1)
            m.next = "IDLE"
    with m.Else():
        m.d.sync += ctr.eq(ctr + 1)
        m.next = loop_back_state


def _build_scalar_fsm(m, stores, result, int_wports, start, done):
    """FSM for scalar kernels (no loops). One or more compute cycles.

    When multiple stores target the same buffer, they are serialized
    across cycles (one write per buffer per cycle).
    """
    waves = _group_stores_by_wave(stores)
    first = "SCALAR" if len(waves) <= 1 else "SCALAR_0"

    with m.FSM(init="IDLE"):
        with m.State("IDLE"):
            m.d.sync += done.eq(0)
            with m.If(start):
                m.next = first

        for wi, wave in enumerate(waves):
            name = "SCALAR" if len(waves) <= 1 else f"SCALAR_{wi}"
            with m.State(name):
                _emit_stores(m, wave, result, int_wports)
                if wi == len(waves) - 1:
                    m.d.sync += done.eq(1)
                    m.next = "IDLE"
                else:
                    m.next = f"SCALAR_{wi + 1}"


def _group_stores_by_wave(stores):
    """Group stores so each wave has at most one IRBufStore per buffer.

    IRRegStores (accumulator writes) are always safe to batch — they all
    target the single acc register.  Conflict only arises for IRBufStore
    nodes targeting the same buf_idx (single write port per buffer).
    """
    if not stores:
        return [stores]
    waves = []
    current_wave = []
    current_bufs = set()
    for store in stores:
        buf = store.buf_idx if isinstance(store, IRBufStore) else None
        if buf is not None and buf in current_bufs:
            waves.append(current_wave)
            current_wave = [store]
            current_bufs = {buf}
        else:
            current_wave.append(store)
            if buf is not None:
                current_bufs.add(buf)
    if current_wave:
        waves.append(current_wave)
    return waves


def _emit_stores(m, stores, result: ArithResult, int_wports: dict):
    """Emit side effects (register writes + memory writes) for a list of stores.

    Called within an FSM state context (inside m.State(...)).
    """
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
            wp = int_wports.get(store.buf_idx)
            if wp is None:
                raise RuntimeError(
                    f"_emit_stores: IRBufStore targets buf_idx={store.buf_idx} "
                    "but no write port exists for that buffer index. "
                    "Check that _create_memories() created a write port for every buffer."
                )
            m.d.comb += [
                wp.addr.eq(addr_sig),
                wp.data.eq(value_sig),
                wp.en.eq(1),
            ]
