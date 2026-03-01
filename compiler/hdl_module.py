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

    def __init__(self, uops, buf_infos):
        self.uops = uops
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

        # --- Pass 1: Build typed IR ---
        from .ir import DType, BufferMeta
        from .uop_to_ir import uop_to_ir
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

        # --- Pass 2: Create counters + build arithmetic datapath ---
        counter_sigs = create_counters(kernel_ir, m)

        acc = None
        if kernel_ir.acc_dtype is not None:
            acc = Signal(kernel_ir.acc_dtype.amaranth_shape(), name="acc")

        arith = ArithmeticLowering(kernel_ir, m, int_rports, counter_sigs, acc)
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
