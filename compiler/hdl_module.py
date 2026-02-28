"""Generated Amaranth module from compiled UOps.

CompiledKernel takes analyzed UOp data and builds an FSM-based
Amaranth module: memories for buffers, counters for loops,
combinational datapath, and FSM for sequencing.

Three-pass architecture:
  Pass 0: Create memories (one per DEFINE_GLOBAL buffer)
  Pass 1: Parse loop structure (UOps → LoopLevel tree)
  Pass 2: Create counters + build combinational datapath
  Pass 3: Wire default write ports + build FSM from loop tree
"""

from amaranth.hdl import Elaboratable, Module, Signal, signed, unsigned, Mux, Const, Cat
from amaranth.lib.memory import Memory

from typing import List
from dataclasses import dataclass, field

from tinygrad import UOp
from tinygrad.uop.ops import Ops, AxisType
from tinygrad.dtype import AddrSpace

@dataclass
class LoopLevel:
  axis_type: AxisType | None  # LOOP, REDUCE, or None (root)
  bound: int                   # how many iterations (0 for root)
  counter: UOp | None          # the RANGE uop itself (None for root)
  prologue: list[UOp]          # ops before inner RANGE (or all ops if innermost)
  body: LoopLevel | None       # nested level, None if innermost
  epilogue: list[UOp]          # ops after inner END

class CompiledKernel(Elaboratable):
    """Hardware module generated from tinygrad UOps.

    Parameters
    ----------
    uops : list[UOp]
        Linearized UOp list from tinygrad.
    buf_infos : list[dict]
        Buffer descriptors: {idx, depth, elem_width, is_signed, is_output}.
    """

    def __init__(self, uops, buf_infos):
        self.uops = uops
        self.buf_infos = buf_infos

        # Control signals
        self.start = Signal()
        self.done = Signal()
        self.busy = Signal()

        # External write ports for loading data into input buffers
        # and read port for reading output buffer
        self.buf_write_ports = {}  # buf_idx → (wen, waddr, wdata)
        self.buf_read_ports = {}  # buf_idx → (raddr, rdata)

        for info in buf_infos:
            idx = info["idx"]
            depth = info["depth"]
            w = info["elem_width"]
            is_signed = info["is_signed"]
            shape = signed(w) if is_signed else unsigned(w)

            self.buf_write_ports[idx] = {
                "wen": Signal(name=f"buf{idx}_wen"),
                "waddr": Signal(range(max(depth, 1)), name=f"buf{idx}_waddr"),
                "wdata": Signal(shape, name=f"buf{idx}_wdata"),
            }

            self.buf_read_ports[idx] = {
                "raddr": Signal(range(max(depth, 1)), name=f"buf{idx}_raddr"),
                "rdata": Signal(shape, name=f"buf{idx}_rdata"),
            }

    def elaborate(self, platform):
        m = Module()

        # --- Pass 0: Create memories ---
        memories, int_rports, int_wports = self._create_memories(m)

        # --- Pass 1: Parse loop structure ---
        root = self._parse_loop_structure(self.uops)

        # --- Pass 2: Create counters + build combinational datapath ---
        counter_map = {}
        acc = Signal(signed(32), name="acc")
        self._create_counters(root, counter_map)
        sig = self._build_datapath(m, self.uops, counter_map, acc, int_rports)

        # --- Pass 3: Wire default write ports + build FSM ---
        self._wire_default_write_ports(m, int_wports)
        self._build_fsm(m, root, sig, acc, counter_map, int_wports)

        return m

    # ------------------------------------------------------------------
    # Pass 0: Memory creation
    # ------------------------------------------------------------------

    def _create_memories(self, m):
        memories = {}
        int_rports = {}
        int_wports = {}

        for info in self.buf_infos:
            idx = info["idx"]
            depth = info["depth"]
            w = info["elem_width"]
            is_signed = info["is_signed"]
            shape = signed(w) if is_signed else unsigned(w)

            mem = Memory(shape=shape, depth=max(depth, 1), init=[0] * max(depth, 1))
            m.submodules[f"buf{idx}"] = mem
            memories[idx] = mem

            # Internal combinational read port (for datapath)
            rp = mem.read_port(domain="comb")
            int_rports[idx] = rp

            # Internal write port (shared between external loading and FSM output)
            wp = mem.write_port()
            int_wports[idx] = wp

            # External read port wiring
            ext_rp = mem.read_port(domain="comb")
            ext = self.buf_read_ports[idx]
            m.d.comb += [
                ext_rp.addr.eq(ext["raddr"]),
                ext["rdata"].eq(ext_rp.data),
            ]

        return memories, int_rports, int_wports

    # ------------------------------------------------------------------
    # Pass 1: Parse loop structure
    # ------------------------------------------------------------------

    def _parse_loop_structure(self, uops: List[UOp]) -> LoopLevel:
        """Parse UOps into a LoopLevel tree.

        Returns the root level (axis_type=None). Root's body is the outermost
        loop, or None for scalar kernels. Pre-loop and post-loop ops are
        captured in root's prologue/epilogue.
        """
        root = LoopLevel(axis_type=None, bound=0, counter=None, prologue=[], body=None,
  epilogue=[])

        stack = [root]
        mode_stack = ["before"]

        to_skip = set([
                Ops.DEFINE_GLOBAL,
                Ops.DEFINE_REG,
                Ops.CONST,
                Ops.SINK,
                Ops.INDEX
                ])

        for u in uops:
            if u.op in to_skip:
                continue
            if u.op == Ops.RANGE:
                to_skip -= set([Ops.INDEX])
                parent = stack[-1]
                child = LoopLevel(
                        axis_type = u.arg[1],
                        bound = u.src[0].arg,
                        body = None,
                        counter = u,
                        epilogue=[],
                        prologue=[]
                        )

                parent.body = child
                stack.append(child)

                mode_stack.append("before")
            elif u.op == Ops.END:
                stack.pop()
                mode_stack.pop()
                if mode_stack:
                    mode_stack[-1] = "after"
            else:
                cur = stack[-1]
                mode = mode_stack[-1]
                if mode == "before":
                    cur.prologue.append(u)
                else:
                    cur.epilogue.append(u)

        return root

    # ------------------------------------------------------------------
    # Pass 2a: Create counter signals from loop tree
    # ------------------------------------------------------------------

    def _create_counters(self, root, counter_map):
        """Walk loop tree, create a counter Signal for each RANGE level."""
        level = root.body
        d = 0
        while level is not None:
            ctr = Signal(range(max(level.bound, 1)), name=f"ctr_L{d}")
            counter_map[id(level.counter)] = ctr
            level = level.body
            d += 1

    # ------------------------------------------------------------------
    # Pass 2b: Build combinational datapath
    # ------------------------------------------------------------------

    def _build_datapath(self, m, uops, counter_map, acc, int_rports):
        """Build combinational datapath from UOps.

        Returns sig dict mapping id(uop) → Amaranth Signal/Value.
        STORE ops map to None — they are resolved at FSM build time.
        """
        sig = {}
        buf_infos = self.buf_infos

        for u in uops:
            if u.op == Ops.DEFINE_GLOBAL:
                sig[id(u)] = None
                continue

            elif u.op == Ops.DEFINE_REG:
                sig[id(u)] = acc
                continue

            elif u.op == Ops.CONST:
                sig[id(u)] = u.arg
                continue

            elif u.op == Ops.RANGE:
                sig[id(u)] = counter_map[id(u)]
                continue

            elif u.op == Ops.AFTER:
                sig[id(u)] = sig.get(id(u.src[0]))
                continue

            elif u.op == Ops.INDEX:
                buf_ptr = u.src[0]
                offset = u.src[1]

                # Register index
                if hasattr(u.dtype, "addrspace") and u.dtype.addrspace == AddrSpace.REG:
                    sig[id(u)] = acc
                    continue

                # Global buffer index — compute address and wire to read port
                buf_idx = self._find_buf_idx(buf_ptr, uops)
                offset_sig = self._to_signal(m, sig, offset, f"idx_{buf_idx}")

                depth = 1
                for info in buf_infos:
                    if info["idx"] == buf_idx:
                        depth = info["depth"]
                        break
                addr_sig = Signal(
                    range(max(depth, 1)), name=f"addr_b{buf_idx}_{id(u) % 10000}"
                )
                m.d.comb += addr_sig.eq(offset_sig)

                sig[id(u)] = ("index", buf_idx, addr_sig)
                continue

            elif u.op == Ops.LOAD:
                index_info = sig.get(id(u.src[0]))

                if index_info is acc or (not isinstance(index_info, tuple)):
                    sig[id(u)] = acc
                    continue

                # Load from memory
                _, buf_idx, addr_sig = index_info
                rp = int_rports[buf_idx]

                m.d.comb += rp.addr.eq(addr_sig)

                info = next(i for i in buf_infos if i["idx"] == buf_idx)
                w = info["elem_width"]
                is_signed = info["is_signed"]
                shape = signed(w) if is_signed else unsigned(w)
                load_sig = Signal(shape, name=f"load_b{buf_idx}_{id(u) % 10000}")
                m.d.comb += load_sig.eq(rp.data)
                sig[id(u)] = load_sig
                continue

            elif u.op == Ops.STORE:
                # Stores are resolved at FSM build time
                sig[id(u)] = None
                continue

            elif u.op == Ops.MUL:
                a = self._to_signal(m, sig, u.src[0], f"mul_a_{id(u) % 10000}")
                b = self._to_signal(m, sig, u.src[1], f"mul_b_{id(u) % 10000}")
                w, is_s = self._dtype_to_width(u.dtype)
                shape = signed(w) if is_s else unsigned(w)
                result = Signal(shape, name=f"mul_{id(u) % 10000}")
                m.d.comb += result.eq(a * b)
                sig[id(u)] = result
                continue

            elif u.op == Ops.ADD:
                a = self._to_signal(m, sig, u.src[0], f"add_a_{id(u) % 10000}")
                b = self._to_signal(m, sig, u.src[1], f"add_b_{id(u) % 10000}")
                w, is_s = self._dtype_to_width(u.dtype)
                shape = signed(w) if is_s else unsigned(w)
                result = Signal(shape, name=f"add_{id(u) % 10000}")
                m.d.comb += result.eq(a + b)
                sig[id(u)] = result
                continue

            elif u.op == Ops.CAST:
                src_val = self._to_signal(m, sig, u.src[0], f"cast_in_{id(u) % 10000}")
                w, is_s = self._dtype_to_width(u.dtype)
                shape = signed(w) if is_s else unsigned(w)
                result = Signal(shape, name=f"cast_{id(u) % 10000}")
                m.d.comb += result.eq(src_val)
                sig[id(u)] = result
                continue

            elif u.op == Ops.CMPLT:
                a = self._to_signal(m, sig, u.src[0], f"cmplt_a_{id(u) % 10000}")
                b = self._to_signal(m, sig, u.src[1], f"cmplt_b_{id(u) % 10000}")
                result = Signal(name=f"cmplt_{id(u) % 10000}")
                m.d.comb += result.eq(a < b)
                sig[id(u)] = result
                continue

            elif u.op == Ops.WHERE:
                cond = self._to_signal(m, sig, u.src[0], f"where_c_{id(u) % 10000}")
                true_val = self._to_signal(m, sig, u.src[1], f"where_t_{id(u) % 10000}")
                false_val = self._to_signal(
                    m, sig, u.src[2], f"where_f_{id(u) % 10000}"
                )
                w, is_s = self._dtype_to_width(u.dtype)
                shape = signed(w) if is_s else unsigned(w)
                result = Signal(shape, name=f"where_{id(u) % 10000}")
                m.d.comb += result.eq(Mux(cond, true_val, false_val))
                sig[id(u)] = result
                continue

            elif u.op == Ops.MAX:
                a = self._to_signal(m, sig, u.src[0], f"max_a_{id(u) % 10000}")
                b = self._to_signal(m, sig, u.src[1], f"max_b_{id(u) % 10000}")
                w, is_s = self._dtype_to_width(u.dtype)
                shape = signed(w) if is_s else unsigned(w)
                result = Signal(shape, name=f"max_{id(u) % 10000}")
                m.d.comb += result.eq(Mux(a > b, a, b))
                sig[id(u)] = result
                continue

            elif u.op in (Ops.END, Ops.SINK):
                sig[id(u)] = None
                continue

            else:
                sig[id(u)] = None
                continue

        return sig

    # ------------------------------------------------------------------
    # Pass 3a: Wire default write ports (external loading)
    # ------------------------------------------------------------------

    def _wire_default_write_ports(self, m, int_wports):
        """Wire external write ports as default drivers for memory write ports.

        FSM states override these with m.d.comb assignments when writing output.
        """
        for info in self.buf_infos:
            idx = info["idx"]
            wp = int_wports[idx]
            ext = self.buf_write_ports[idx]
            m.d.comb += [
                wp.addr.eq(ext["waddr"]),
                wp.data.eq(ext["wdata"]),
                wp.en.eq(ext["wen"]),
            ]

    # ------------------------------------------------------------------
    # Pass 3b: Build FSM from loop tree
    # ------------------------------------------------------------------

    def _build_fsm(self, m, root, sig, acc, counter_map, int_wports):
        """Build FSM that sequences stores according to the loop structure.

        State naming:
          IDLE         — wait for start
          L{d}_PRO     — non-innermost level prologue (e.g. acc reset)
          L{d}_BODY    — innermost level body (e.g. MAC)
          L{d}_EPI     — non-innermost level epilogue (e.g. output write)

        Done signal is registered (m.d.sync) and set in the last compute
        state when the outermost counter reaches its bound.
        """
        # Flatten loop levels
        levels = []  # [(LoopLevel, depth)]
        level = root.body
        d = 0
        while level is not None:
            levels.append((level, d))
            level = level.body
            d += 1

        # Root-level stores (for scalar kernels with no loops)
        root_stores = [u for u in root.prologue + root.epilogue if u.op == Ops.STORE]

        if not levels:
            self._build_scalar_fsm(m, root_stores, sig, acc, int_wports)
            return

        # Determine first state after IDLE
        if levels[0][0].body is None:
            first_state = "L0_BODY"
        else:
            first_state = "L0_PRO"

        with m.FSM(init="IDLE"):
            with m.State("IDLE"):
                m.d.sync += self.done.eq(0)
                with m.If(self.start):
                    ctr = counter_map[id(levels[0][0].counter)]
                    m.d.sync += ctr.eq(0)
                    m.next = first_state

            for i, (level, d) in enumerate(levels):
                is_innermost = (level.body is None)
                pro_stores = [u for u in level.prologue if u.op == Ops.STORE]
                epi_stores = [u for u in level.epilogue if u.op == Ops.STORE]
                ctr = counter_map[id(level.counter)]

                if is_innermost:
                    with m.State(f"L{d}_BODY"):
                        self._emit_stores(m, sig, pro_stores, acc, int_wports)
                        with m.If(ctr == level.bound - 1):
                            if d > 0:
                                m.next = f"L{d-1}_EPI"
                            else:
                                # Single-level loop, outermost is innermost
                                m.d.sync += self.done.eq(1)
                                m.next = "IDLE"
                        with m.Else():
                            m.d.sync += ctr.eq(ctr + 1)
                            m.next = f"L{d}_BODY"
                else:
                    # Non-innermost: PRO and EPI states
                    child_level, child_d = levels[i + 1]
                    if child_level.body is None:
                        child_first = f"L{child_d}_BODY"
                    else:
                        child_first = f"L{child_d}_PRO"

                    with m.State(f"L{d}_PRO"):
                        self._emit_stores(m, sig, pro_stores, acc, int_wports)
                        child_ctr = counter_map[id(child_level.counter)]
                        m.d.sync += child_ctr.eq(0)
                        m.next = child_first

                    with m.State(f"L{d}_EPI"):
                        self._emit_stores(m, sig, epi_stores, acc, int_wports)
                        with m.If(ctr == level.bound - 1):
                            if d > 0:
                                m.next = f"L{d-1}_EPI"
                            else:
                                m.d.sync += self.done.eq(1)
                                m.next = "IDLE"
                        with m.Else():
                            m.d.sync += ctr.eq(ctr + 1)
                            m.next = f"L{d}_PRO"

    def _build_scalar_fsm(self, m, stores, sig, acc, int_wports):
        """FSM for scalar kernels (no loops). One compute cycle."""
        with m.FSM(init="IDLE"):
            with m.State("IDLE"):
                m.d.sync += self.done.eq(0)
                with m.If(self.start):
                    m.next = "SCALAR"
            with m.State("SCALAR"):
                self._emit_stores(m, sig, stores, acc, int_wports)
                m.d.sync += self.done.eq(1)
                m.next = "IDLE"

    # ------------------------------------------------------------------
    # Store emission helper
    # ------------------------------------------------------------------

    def _emit_stores(self, m, sig, stores, acc, int_wports):
        """Emit STORE ops within the current FSM state context.

        Resolves store targets by looking up sig[id(store.src[0])]:
          - If target is acc (register) → m.d.sync += acc.eq(value)
          - If target is ("index", buf_idx, addr) → memory write via comb override
        """
        for store_uop in stores:
            index_info = sig.get(id(store_uop.src[0]))
            value_sig = self._to_signal(
                m, sig, store_uop.src[1], f"sv_{id(store_uop) % 10000}"
            )

            if index_info is acc:
                # Register store (accumulator update)
                m.d.sync += acc.eq(value_sig)
            elif isinstance(index_info, tuple) and index_info[0] == "index":
                # Memory store — override default write port wiring
                _, buf_idx, addr_sig = index_info
                wp = int_wports[buf_idx]
                m.d.comb += [
                    wp.addr.eq(addr_sig),
                    wp.data.eq(value_sig),
                    wp.en.eq(1),
                ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_buf_idx(self, ptr_uop, uops):
        """Walk through AFTER chains to find the DEFINE_GLOBAL buffer index."""
        current = ptr_uop
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if current.op == Ops.DEFINE_GLOBAL:
                return current.arg
            if current.op == Ops.DEFINE_REG:
                return -1  # register, not a buffer
            if current.op == Ops.AFTER:
                current = current.src[0]
                continue
            break
        return -1

    def _to_signal(self, m, sig, uop, name):
        """Convert a UOp reference to an Amaranth Signal/Value."""
        val = sig.get(id(uop))
        if val is None:
            return Const(0)
        if isinstance(val, int):
            return val
        if isinstance(val, tuple):
            # Shouldn't happen for values, but handle gracefully
            return Const(0)
        return val

    def _dtype_to_width(self, dtype):
        """Return (bit_width, is_signed) for a tinygrad dtype."""
        from tinygrad import dtypes

        if dtype == dtypes.char or dtype == dtypes.int8:
            return 8, True
        if dtype == dtypes.uchar or dtype == dtypes.uint8:
            return 8, False
        if dtype == dtypes.short or dtype == dtypes.int16:
            return 16, True
        if dtype == dtypes.int or dtype == dtypes.int32:
            return 32, True
        if dtype == dtypes.uint or dtype == dtypes.uint32:
            return 32, False
        if dtype == dtypes.bool:
            return 1, False
        # Default to 32-bit signed
        return 32, True
