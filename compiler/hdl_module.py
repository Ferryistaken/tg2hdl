"""Generated Amaranth module from compiled UOps.

CompiledKernel takes analyzed UOp data and builds an FSM-based
Amaranth module: memories for buffers, counters for loops,
combinational datapath, and FSM for sequencing.

Four-pass architecture:
  Pass 0: Create memories (one per DEFINE_GLOBAL buffer)
  Pass 1: Build typed KernelIR via uop_to_ir()
  Pass 2: Create counters + arithmetic datapath via ArithmeticLowering
  Pass 3: Wire default write ports + build FSM via build_control()
"""

from amaranth.hdl import Elaboratable, Module, Signal, signed, unsigned
from amaranth.lib.memory import Memory


class CompiledKernel(Elaboratable):
    """Hardware module generated from tinygrad UOps.

    Parameters
    ----------
    uops : list[UOp]
        Linearized UOp list from tinygrad.
    buf_infos : list[dict]
        Buffer descriptors: {idx, depth, elem_width, is_signed, is_output}.
    """

    def __init__(self, uops, buf_infos, compile_options=None):
        self.uops = uops
        self.buf_infos = buf_infos
        self.compile_options = compile_options
        self.compile_report = {}
        self._active_unroll = 1
        self._read_port_budget = {}
        self._write_port_budget = {}

        # Control signals
        self.start = Signal()
        self.done = Signal()
        self.busy = Signal()

        # External write ports for loading data into input buffers
        # and read port for reading output buffer
        self.buf_write_ports = {}  # buf_idx → {wen, waddr, wdata}
        self.buf_read_ports = {}   # buf_idx → {raddr, rdata}

        for info in buf_infos:
            idx = info["idx"]
            depth = info["depth"]
            w = info["elem_width"]
            is_signed = info["is_signed"]
            shape = signed(w) if is_signed else unsigned(w)

            self.buf_write_ports[idx] = {
                "wen":   Signal(name=f"buf{idx}_wen"),
                "waddr": Signal(range(max(depth, 1)), name=f"buf{idx}_waddr"),
                "wdata": Signal(shape, name=f"buf{idx}_wdata"),
            }

            self.buf_read_ports[idx] = {
                "raddr": Signal(range(max(depth, 1)), name=f"buf{idx}_raddr"),
                "rdata": Signal(shape, name=f"buf{idx}_rdata"),
            }

    def elaborate(self, platform):
        m = Module()

        # --- Pass 0: Build typed IR + transforms ---
        from .ir import DType, BufferMeta
        from .uop_to_ir import uop_to_ir
        from .transforms import apply_loop_unroll
        from .lowering.arithmetic import ArithmeticLowering, create_counters
        from .lowering.control import build_control

        buf_metas = [
            BufferMeta(
                idx=b["idx"],
                depth=b["depth"],
                dtype=DType.from_width(b["elem_width"], b["is_signed"]),
                is_output=b["is_output"],
            )
            for b in self.buf_infos
        ]
        kernel_ir = uop_to_ir(self.uops, buf_metas)
        unroll_requested = getattr(self.compile_options, "unroll_loop", 1)
        kernel_ir, unroll_info = apply_loop_unroll(kernel_ir, unroll_requested)
        self.compile_report = {
            "unroll_loop_requested": unroll_info.requested_unroll,
            "unroll_loop_applied": unroll_info.applied_unroll,
            "loop_bound": unroll_info.loop_bound,
            "loop_tail": unroll_info.tail_length,
            "loop_iterations_per_cycle_est": unroll_info.iterations_per_cycle_est,
            "unroll_fallback_reason": unroll_info.fallback_reason,
        }
        self._active_unroll = unroll_info.applied_unroll
        self._read_port_budget, self._write_port_budget = self._estimate_port_budgets(kernel_ir)

        # --- Pass 1: Create memories/ports ---
        memories, int_rports, int_wports = self._create_memories(m)

        # --- Pass 2: Create counters + build arithmetic datapath ---
        counter_sigs = create_counters(kernel_ir, m)

        acc = None
        if kernel_ir.acc_dtype is not None:
            acc = Signal(kernel_ir.acc_dtype.amaranth_shape(), name="acc")

        arith = ArithmeticLowering(
            kernel_ir,
            m,
            int_rports,
            counter_sigs,
            acc,
            unroll_factor=self._active_unroll,
        )
        arith_result = arith.run()

        # --- Pass 3: Wire default write ports + build FSM ---
        self._wire_default_write_ports(m, int_wports)
        build_control(m, kernel_ir, arith_result, int_wports,
                      self.start, self.done, self.busy)

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

            # Internal combinational read ports (one per active lane)
            lane_reads = []
            for _ in range(max(self._read_port_budget.get(idx, 1), 1)):
                rp = mem.read_port(domain="comb")
                lane_reads.append(rp)
            int_rports[idx] = lane_reads

            # Internal write ports (external loading uses port 0)
            lane_writes = []
            for _ in range(max(self._write_port_budget.get(idx, 1), 1)):
                wp = mem.write_port()
                lane_writes.append(wp)
            int_wports[idx] = lane_writes

            # External read port wiring
            ext_rp = mem.read_port(domain="comb")
            ext = self.buf_read_ports[idx]
            m.d.comb += [
                ext_rp.addr.eq(ext["raddr"]),
                ext["rdata"].eq(ext_rp.data),
            ]

        return memories, int_rports, int_wports

    # ------------------------------------------------------------------
    # Pass 3a: Wire default write ports (external loading)
    # ------------------------------------------------------------------

    def _wire_default_write_ports(self, m, int_wports):
        """Wire external write ports as default drivers for memory write ports.

        FSM states override these with m.d.comb assignments when writing output.
        """
        for info in self.buf_infos:
            idx = info["idx"]
            wp = int_wports[idx][0]
            ext = self.buf_write_ports[idx]
            m.d.comb += [
                wp.addr.eq(ext["waddr"]),
                wp.data.eq(ext["wdata"]),
                wp.en.eq(ext["wen"]),
            ]
            for extra_wp in int_wports[idx][1:]:
                m.d.comb += extra_wp.en.eq(0)

    def _estimate_port_budgets(self, kernel_ir):
        from .ir import IRBufLoad, IRBufStore, IROp, IRRegStore

        read_nodes = {}
        write_budget = {b["idx"]: 1 for b in self.buf_infos}

        def collect_val(v):
            if isinstance(v, IRBufLoad):
                read_nodes.setdefault(v.buf_idx, set()).add(id(v))
                collect_val(v.addr)
            elif isinstance(v, IROp):
                for s in v.srcs:
                    collect_val(s)

        def consume_store_list(stores):
            per_buf_writes = {}
            for s in stores:
                if isinstance(s, IRBufStore):
                    per_buf_writes[s.buf_idx] = per_buf_writes.get(s.buf_idx, 0) + 1
                    collect_val(s.addr)
                    collect_val(s.value)
                elif isinstance(s, IRRegStore):
                    collect_val(s.value)
            for buf_idx, cnt in per_buf_writes.items():
                write_budget[buf_idx] = max(write_budget.get(buf_idx, 1), cnt)

        levels = []
        level = kernel_ir.loop_tree.body
        while level is not None:
            levels.append(level)
            level = level.body

        if not levels:
            root_scalar = kernel_ir.scalar_stores + [
                s for s in (kernel_ir.loop_tree.prologue + kernel_ir.loop_tree.epilogue)
                if isinstance(s, IRBufStore)
            ]
            consume_store_list(root_scalar)
        else:
            for lvl in levels:
                consume_store_list(lvl.prologue)
                consume_store_list(lvl.epilogue)

        read_budget = {b["idx"]: 1 for b in self.buf_infos}
        for buf_idx, nodes in read_nodes.items():
            read_budget[buf_idx] = max(read_budget.get(buf_idx, 1), len(nodes))

        return read_budget, write_budget
