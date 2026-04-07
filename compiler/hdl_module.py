"""Generated Amaranth module from KernelIR.

CompiledKernel takes a fully-built KernelIR and lowers it to an
FSM-based Amaranth module: memories for buffers, counters for loops,
combinational datapath, and FSM for sequencing.

Three-pass architecture:
  Pass 0: Create memories (one per DEFINE_GLOBAL buffer)
  Pass 1: Create counters + arithmetic datapath via ArithmeticLowering
  Pass 2: Wire default write ports + build FSM via build_control()
"""

from amaranth.hdl import Elaboratable, Module, Signal, signed, unsigned
from amaranth.lib.memory import Memory


class CompiledKernel(Elaboratable):
    """Hardware module generated from KernelIR.

    Parameters
    ----------
    kernel_ir : KernelIR
        Typed kernel IR produced by uop_to_ir().
    buf_infos : list[dict]
        Buffer descriptors: {idx, depth, elem_width, is_signed, is_output}.
    """

    def __init__(self, kernel_ir, buf_infos):
        self.kernel_ir = kernel_ir
        self.buf_infos = buf_infos

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

        # --- Pass 0: Create memories ---
        memories, int_rports, int_wports = self._create_memories(m)

        from .lowering.arithmetic import ArithmeticLowering, create_counters
        from .lowering.control import build_control

        kernel_ir = self.kernel_ir

        # --- Pass 1: Create counters + build arithmetic datapath ---
        counter_sigs = create_counters(kernel_ir, m)

        acc = None
        if kernel_ir.acc_dtype is not None:
            acc = Signal(kernel_ir.acc_dtype.amaranth_shape(), name="acc")

        arith = ArithmeticLowering(kernel_ir, m, int_rports, counter_sigs, acc)
        arith_result = arith.run()

        # --- Pass 2: Wire default write ports + build FSM ---
        self._wire_default_write_ports(m, int_wports)
        build_control(m, kernel_ir, arith_result, int_wports,
                      self.start, self.done, self.busy)

        return m

    # ------------------------------------------------------------------
    # Pass 0: Memory creation
    # ------------------------------------------------------------------

    def _create_memories(self, m):
        memories = {}
        int_rports = {}   # buf_idx → list[read_port]
        int_wports = {}

        # Count how many read ports each buffer needs
        load_counts = self._count_loads_per_buffer()

        for info in self.buf_infos:
            idx = info["idx"]
            depth = info["depth"]
            w = info["elem_width"]
            is_signed = info["is_signed"]
            shape = signed(w) if is_signed else unsigned(w)

            mem = Memory(shape=shape, depth=max(depth, 1), init=[0] * max(depth, 1))
            m.submodules[f"buf{idx}"] = mem
            memories[idx] = mem

            # Internal combinational read ports (for datapath)
            n_ports = max(load_counts.get(idx, 1), 1)
            ports = []
            for p in range(n_ports):
                rp = mem.read_port(domain="comb")
                ports.append(rp)
            int_rports[idx] = ports

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

    def _count_loads_per_buffer(self):
        """Count distinct IRBufLoad nodes per buffer index in the KernelIR."""
        from .ir import IRBufLoad, IRBufStore, IRRegStore, IROp

        counts = {}
        visited = set()

        def walk(val):
            if val is None or id(val) in visited:
                return
            visited.add(id(val))
            if isinstance(val, IRBufLoad):
                counts[val.buf_idx] = counts.get(val.buf_idx, 0) + 1
                walk(val.addr)
            elif isinstance(val, IROp):
                for s in val.srcs:
                    walk(s)

        def walk_loop(node):
            for store in node.prologue + node.epilogue:
                if isinstance(store, (IRBufStore, IRRegStore)):
                    walk(store.value)
                if isinstance(store, IRBufStore):
                    walk(store.addr)
            if node.body is not None:
                walk_loop(node.body)

        walk_loop(self.kernel_ir.loop_tree)
        for s in self.kernel_ir.scalar_stores:
            walk(s.value)
            walk(s.addr)

        return counts

    # ------------------------------------------------------------------
    # Pass 2a: Wire default write ports (external loading)
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
