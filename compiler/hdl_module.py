"""Generated Amaranth module from compiled UOps.

CompiledKernel takes analyzed UOp data and builds an FSM-based
Amaranth module: memories for buffers, counters for loops,
combinational datapath, and FSM for sequencing.
"""

from amaranth.hdl import Elaboratable, Module, Signal, signed, unsigned, Mux, Const, Cat
from amaranth.lib.memory import Memory

from tinygrad.uop.ops import Ops, AxisType  # noqa: F401
from tinygrad.dtype import AddrSpace


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

            # Every buffer gets an external write port for loading
            self.buf_write_ports[idx] = {
                "wen": Signal(name=f"buf{idx}_wen"),
                "waddr": Signal(range(max(depth, 1)), name=f"buf{idx}_waddr"),
                "wdata": Signal(shape, name=f"buf{idx}_wdata"),
            }

            # Every buffer gets an external read port for inspection
            self.buf_read_ports[idx] = {
                "raddr": Signal(range(max(depth, 1)), name=f"buf{idx}_raddr"),
                "rdata": Signal(shape, name=f"buf{idx}_rdata"),
            }

    def elaborate(self, platform):
        m = Module()

        uops = self.uops
        buf_infos = self.buf_infos

        # --- Phase 1: Create memories ---
        memories = {}  # buf_idx → Memory submodule
        int_rports = {}  # buf_idx → internal read port (comb)
        int_wports = {}  # buf_idx → internal write port (sync, for output)

        for info in buf_infos:
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

        # --- Phase 2: Analyze UOp structure ---
        # Find loops (RANGE/END pairs)
        loops = []  # List of (axis_id, axis_type, bound) for each RANGE
        phases = {}  # UOp → phase string

        for u in uops:
            if u.op == Ops.RANGE:
                axis_id, axis_type = u.arg
                bound = u.src[0].arg
                loops.append((axis_id, axis_type, bound))

        # Determine kernel type
        has_reduce = any(at == AxisType.REDUCE for _, at, _ in loops)
        loop_bound = loops[0][2] if loops else 0  # bound for simple loop

        # For GEMV-style kernels
        outer_bound = 0
        inner_bound = 0
        if has_reduce and len(loops) >= 2:
            outer_bound = loops[0][2]
            inner_bound = loops[1][2]

        # Assign phases based on kernel type
        if has_reduce:
            # GEMV-style: INIT_ROW | COMPUTE | POST
            current_phase = "SETUP"
            for u in uops:
                if u.op == Ops.RANGE:
                    _, axis_type = u.arg
                    if axis_type == AxisType.LOOP:
                        current_phase = "INIT_ROW"
                    elif axis_type == AxisType.REDUCE:
                        current_phase = "COMPUTE"
                    phases[id(u)] = current_phase
                    continue
                elif u.op == Ops.END:
                    phases[id(u)] = current_phase
                    end_range = u.src[-1]
                    _, axis_type = end_range.arg
                    if axis_type == AxisType.REDUCE:
                        current_phase = "POST"
                    elif axis_type == AxisType.LOOP:
                        current_phase = "DONE"
                    continue
                elif u.op == Ops.SINK:
                    phases[id(u)] = "DONE"
                    continue
                phases[id(u)] = current_phase
        elif loops:
            # Simple element-wise loop: COMPUTE only
            in_loop = False
            for u in uops:
                if u.op == Ops.RANGE:
                    in_loop = True
                    phases[id(u)] = "COMPUTE"
                    continue
                elif u.op == Ops.END:
                    phases[id(u)] = "COMPUTE"
                    in_loop = False
                    continue
                elif u.op == Ops.SINK:
                    phases[id(u)] = "DONE"
                    continue
                phases[id(u)] = "COMPUTE" if in_loop else "SETUP"
        else:
            # Scalar kernel (no loops at all): everything is COMPUTE
            for u in uops:
                if u.op == Ops.SINK:
                    phases[id(u)] = "DONE"
                else:
                    phases[id(u)] = "COMPUTE"

        # --- Phase 3: Create signals ---
        # Counter signal for loop
        loop_ctr = Signal(range(max(loop_bound, 1)), name="loop_ctr")

        # For GEMV-style kernels
        outer_ctr = Signal(range(max(outer_bound, 1)), name="outer_ctr")
        inner_ctr = Signal(range(max(inner_bound, 1)), name="inner_ctr")

        # Accumulator (only needed for reductions)
        acc = Signal(signed(32), name="acc")

        # Signal map: UOp → Amaranth Signal/Value
        sig = {}

        # --- Phase 4: Build combinational datapath ---
        # Walk all UOps and create their signal equivalents
        for u in uops:
            if u.op == Ops.DEFINE_GLOBAL:
                # Buffer pointer — no signal needed, handled by memory
                sig[id(u)] = None
                continue

            elif u.op == Ops.DEFINE_REG:
                # Register file — maps to accumulator
                sig[id(u)] = acc
                continue

            elif u.op == Ops.CONST:
                sig[id(u)] = u.arg
                continue

            elif u.op == Ops.RANGE:
                # Loop variable — maps to counter
                _, axis_type = u.arg
                if not has_reduce:
                    # Simple loop
                    sig[id(u)] = loop_ctr
                elif axis_type == AxisType.LOOP:
                    sig[id(u)] = outer_ctr
                else:
                    sig[id(u)] = inner_ctr
                continue

            elif u.op == Ops.AFTER:
                # Ordering barrier — pass through first src
                sig[id(u)] = sig.get(id(u.src[0]))
                continue

            elif u.op == Ops.INDEX:
                # Address computation
                buf_ptr = u.src[0]
                offset = u.src[1]

                # Check if this indexes into a REG
                if hasattr(u.dtype, "addrspace") and u.dtype.addrspace == AddrSpace.REG:
                    # Register index — just reference the accumulator
                    sig[id(u)] = acc
                    continue

                # Global buffer index — compute address and wire to read port
                buf_idx = self._find_buf_idx(buf_ptr, uops)
                offset_sig = self._to_signal(m, sig, offset, f"idx_{buf_idx}")

                # Create a signal for the computed address
                depth = 1
                for info in buf_infos:
                    if info["idx"] == buf_idx:
                        depth = info["depth"]
                        break
                addr_sig = Signal(
                    range(max(depth, 1)), name=f"addr_b{buf_idx}_{id(u) % 10000}"
                )
                m.d.comb += addr_sig.eq(offset_sig)

                # Store (buf_idx, addr_sig) tuple
                sig[id(u)] = ("index", buf_idx, addr_sig)
                continue

            elif u.op == Ops.LOAD:
                index_info = sig.get(id(u.src[0]))

                if index_info is acc or (not isinstance(index_info, tuple)):
                    # Load from register → just use accumulator value
                    sig[id(u)] = acc
                    continue

                # Load from memory
                _, buf_idx, addr_sig = index_info
                rp = int_rports[buf_idx]

                # Wire address (will be overwritten if multiple loads use same port,
                # but Amaranth handles last-writer-wins in comb domain)
                m.d.comb += rp.addr.eq(addr_sig)

                # Create a signal for the loaded value
                info = next(i for i in buf_infos if i["idx"] == buf_idx)
                w = info["elem_width"]
                is_signed = info["is_signed"]
                shape = signed(w) if is_signed else unsigned(w)
                load_sig = Signal(shape, name=f"load_b{buf_idx}_{id(u) % 10000}")
                m.d.comb += load_sig.eq(rp.data)
                sig[id(u)] = load_sig
                continue

            elif u.op == Ops.STORE:
                # Stores are handled by the FSM (sequential)
                # Just record what value goes where
                index_info = sig.get(id(u.src[0]))
                value = self._to_signal(m, sig, u.src[1], f"store_val_{id(u) % 10000}")

                if index_info is acc or (not isinstance(index_info, tuple)):
                    # Store to register (accumulator)
                    sig[id(u)] = ("store_reg", value)
                else:
                    # Store to memory
                    _, buf_idx, addr_sig = index_info
                    sig[id(u)] = ("store_mem", buf_idx, addr_sig, value)
                continue

            elif u.op == Ops.MUL:
                a = self._to_signal(m, sig, u.src[0], f"mul_a_{id(u) % 10000}")
                b = self._to_signal(m, sig, u.src[1], f"mul_b_{id(u) % 10000}")
                # Determine result width from dtype
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
                # Unknown op — skip with a warning
                sig[id(u)] = None
                continue

        # --- Phase 5: Build FSM ---
        # Collect stores by phase
        init_row_stores = []  # (target, value) for accumulator resets
        compute_stores = []  # (target, value) for MAC updates
        post_stores = []  # (buf_idx, addr, value) for output writes
        loop_stores = []  # (buf_idx, addr, value) for simple loop stores

        for u in uops:
            if u.op != Ops.STORE:
                continue
            phase = phases.get(id(u), "SETUP")
            store_info = sig.get(id(u))
            if store_info is None:
                continue

            if isinstance(store_info, tuple) and store_info[0] == "store_reg":
                _, value = store_info
                if phase == "INIT_ROW":
                    init_row_stores.append(value)
                elif phase == "COMPUTE":
                    compute_stores.append(value)
            elif isinstance(store_info, tuple) and store_info[0] == "store_mem":
                _, buf_idx, addr_sig, value = store_info
                if phase == "POST":
                    post_stores.append((buf_idx, addr_sig, value))
                elif phase == "COMPUTE" and not has_reduce:
                    # Element-wise (looped or scalar): store in COMPUTE phase
                    loop_stores.append((buf_idx, addr_sig, value))

        # Default: wire external write ports when not computing
        # During computation, output memory write is controlled by FSM
        output_buf_idxs = {s[0] for s in post_stores} | {s[0] for s in loop_stores}

        for info in buf_infos:
            idx = info["idx"]
            wp = int_wports[idx]
            ext = self.buf_write_ports[idx]

            if idx in output_buf_idxs:
                # Output buffer: FSM controls writes during POST,
                # external port controls during IDLE
                # Default to external (overridden in FSM states)
                m.d.comb += [
                    wp.addr.eq(ext["waddr"]),
                    wp.data.eq(ext["wdata"]),
                    wp.en.eq(ext["wen"]),
                ]
            else:
                # Input buffer: always externally controlled
                m.d.comb += [
                    wp.addr.eq(ext["waddr"]),
                    wp.data.eq(ext["wdata"]),
                    wp.en.eq(ext["wen"]),
                ]

        # Build FSM
        if not loops:
            # Scalar kernel (no loops): single-cycle compute
            with m.FSM(init="IDLE"):
                with m.State("IDLE"):
                    with m.If(self.start):
                        m.next = "COMPUTE"

                with m.State("COMPUTE"):
                    m.d.comb += self.busy.eq(1)

                    for buf_idx, addr_sig, value in loop_stores:
                        wp = int_wports[buf_idx]
                        m.d.comb += [
                            wp.en.eq(1),
                            wp.addr.eq(addr_sig),
                            wp.data.eq(value),
                        ]

                    m.next = "DONE"

                with m.State("DONE"):
                    m.d.comb += self.done.eq(1)
                    m.next = "IDLE"

        elif not has_reduce:
            # Simple element-wise loop FSM
            with m.FSM(init="IDLE"):
                with m.State("IDLE"):
                    with m.If(self.start):
                        m.d.sync += loop_ctr.eq(0)
                        m.next = "COMPUTE"

                with m.State("COMPUTE"):
                    m.d.comb += self.busy.eq(1)

                    # Write results to output memory
                    for buf_idx, addr_sig, value in loop_stores:
                        wp = int_wports[buf_idx]
                        m.d.comb += [
                            wp.en.eq(1),
                            wp.addr.eq(addr_sig),
                            wp.data.eq(value),
                        ]

                    with m.If(loop_ctr == loop_bound - 1):
                        m.next = "DONE"
                    with m.Else():
                        m.d.sync += loop_ctr.eq(loop_ctr + 1)

                with m.State("DONE"):
                    m.d.comb += self.done.eq(1)
                    m.next = "IDLE"
        else:
            # GEMV-style kernel FSM
            with m.FSM(init="IDLE"):
                with m.State("IDLE"):
                    with m.If(self.start):
                        m.d.sync += [
                            outer_ctr.eq(0),
                            inner_ctr.eq(0),
                            acc.eq(0),
                        ]
                        m.next = "INIT_ROW"

                with m.State("INIT_ROW"):
                    m.d.comb += self.busy.eq(1)
                    m.d.sync += acc.eq(0)
                    m.d.sync += inner_ctr.eq(0)
                    m.next = "COMPUTE"

                with m.State("COMPUTE"):
                    m.d.comb += self.busy.eq(1)

                    if compute_stores:
                        m.d.sync += acc.eq(compute_stores[-1])

                    with m.If(inner_ctr == inner_bound - 1):
                        m.next = "POST"
                    with m.Else():
                        m.d.sync += inner_ctr.eq(inner_ctr + 1)

                with m.State("POST"):
                    m.d.comb += self.busy.eq(1)

                    for buf_idx, addr_sig, value in post_stores:
                        wp = int_wports[buf_idx]
                        m.d.comb += [
                            wp.en.eq(1),
                            wp.addr.eq(addr_sig),
                            wp.data.eq(value),
                        ]

                    with m.If(outer_ctr == outer_bound - 1):
                        m.next = "DONE"
                    with m.Else():
                        m.d.sync += outer_ctr.eq(outer_ctr + 1)
                        m.next = "INIT_ROW"

                with m.State("DONE"):
                    m.d.comb += self.done.eq(1)
                    m.next = "IDLE"

        return m

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
