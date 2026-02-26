"""INT8 GEMV (General Matrix-Vector Multiply) block in Amaranth HDL.

Computes y[i] = sum_j(W[i][j] * x[j]) for each output element i.

Inputs:  INT8 weight matrix (M x K), INT8 vector (K elements)
Outputs: INT32 result vector (M elements)

FSM: IDLE -> COMPUTE -> EMIT (per row) -> DONE
"""

from amaranth.hdl import Elaboratable, Module, Signal, signed
from amaranth.lib.memory import Memory


class GEMVUnit(Elaboratable):
    """Sequential MAC-based GEMV unit.

    Parameters
    ----------
    m_dim : int
        Number of output elements (rows of weight matrix).
    k_dim : int
        Reduction dimension (columns of weight matrix / vector length).
    """

    def __init__(self, m_dim, k_dim):
        self.m_dim = m_dim
        self.k_dim = k_dim

        # Control
        self.start = Signal()
        self.done = Signal()
        self.busy = Signal()

        # Vector load port
        self.vec_wen = Signal()
        self.vec_waddr = Signal(range(k_dim))
        self.vec_wdata = Signal(signed(8))

        # Weight load port
        self.w_wen = Signal()
        self.w_waddr = Signal(range(m_dim * k_dim))
        self.w_wdata = Signal(signed(8))

        # Result output (active during EMIT state)
        self.result_valid = Signal()
        self.result_idx = Signal(range(m_dim))
        self.result_data = Signal(signed(32))

    def elaborate(self, platform):
        m = Module()

        M = self.m_dim
        K = self.k_dim

        # --- Memories ---
        m.submodules.vec_mem = vec_mem = Memory(
            shape=signed(8), depth=K, init=[0] * K
        )
        m.submodules.w_mem = w_mem = Memory(
            shape=signed(8), depth=M * K, init=[0] * (M * K)
        )

        vec_wp = vec_mem.write_port()
        vec_rp = vec_mem.read_port(domain="comb")
        w_wp = w_mem.write_port()
        w_rp = w_mem.read_port(domain="comb")

        # --- Internal signals ---
        acc = Signal(signed(32))
        row_idx = Signal(range(M))
        col_idx = Signal(range(K))
        product = Signal(signed(16))

        m.d.comb += product.eq(w_rp.data * vec_rp.data)

        # --- Load wiring (active in any state) ---
        m.d.comb += [
            vec_wp.addr.eq(self.vec_waddr),
            vec_wp.data.eq(self.vec_wdata),
            vec_wp.en.eq(self.vec_wen),
            w_wp.addr.eq(self.w_waddr),
            w_wp.data.eq(self.w_wdata),
            w_wp.en.eq(self.w_wen),
        ]

        # --- FSM ---
        with m.FSM(init="IDLE"):
            with m.State("IDLE"):
                with m.If(self.start):
                    m.d.sync += [
                        row_idx.eq(0),
                        col_idx.eq(0),
                        acc.eq(0),
                    ]
                    m.next = "COMPUTE"

            with m.State("COMPUTE"):
                m.d.comb += self.busy.eq(1)

                # Read W[row_idx][col_idx] and x[col_idx]
                m.d.comb += [
                    vec_rp.addr.eq(col_idx),
                    w_rp.addr.eq(row_idx * K + col_idx),
                ]

                # Accumulate
                m.d.sync += acc.eq(acc + product)

                with m.If(col_idx == K - 1):
                    m.next = "EMIT"
                with m.Else():
                    m.d.sync += col_idx.eq(col_idx + 1)

            with m.State("EMIT"):
                # acc now holds the complete dot product for this row
                m.d.comb += [
                    self.busy.eq(1),
                    self.result_valid.eq(1),
                    self.result_data.eq(acc),
                    self.result_idx.eq(row_idx),
                ]

                m.d.sync += [
                    acc.eq(0),
                    col_idx.eq(0),
                ]

                with m.If(row_idx == M - 1):
                    m.next = "DONE"
                with m.Else():
                    m.d.sync += row_idx.eq(row_idx + 1)
                    m.next = "COMPUTE"

            with m.State("DONE"):
                m.d.comb += self.done.eq(1)
                m.next = "IDLE"

        return m
