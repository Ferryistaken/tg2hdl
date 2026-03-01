"""TopModule: sequences N CompiledKernels with a hardware copy FSM.

TopModule wires multiple CompiledKernel instances into a single Amaranth
Elaboratable. A copy FSM transfers each kernel's output into the next
kernel's input buffer between compute phases.

Ports
-----
start : Signal, in
    Pulse high for one cycle to begin execution.
done : Signal, out
    Pulses high for one cycle when all kernels have finished.
ext_write_ports : dict[(k_idx, buf_idx), {wen, waddr, wdata}]
    External write ports for all non-intermediate input buffers.
output_rport : {raddr, rdata}
    Read port for the final kernel's output buffer.

FSM sequence (two-kernel example)
-----------------------------------
IDLE → K0_RUN → K0_WAIT → COPY_0_1 → K1_RUN → K1_WAIT → DONE → IDLE
"""

import time
from dataclasses import dataclass

import numpy as np
from amaranth.hdl import Elaboratable, Module, Signal, signed, unsigned
from amaranth.sim import Simulator

from .hdl_module import CompiledKernel


class TopModule(Elaboratable):
    """Hardware sequencer for N compiled kernels.

    Parameters
    ----------
    kernels : list[CompiledKernel]
        Kernels to run in order.
    connections : list[tuple[int, int, int, int]]
        Each entry is (src_k, src_buf, dst_k, dst_buf):
        the output buf *src_buf* of kernel *src_k* is DMA-copied into
        input buf *dst_buf* of kernel *dst_k* between compute phases.
        For a simple chain this would be [(0, 0, 1, j)] for each input j
        of kernel 1 that came from kernel 0's output.
    buf_depths : dict[tuple[int,int], int]
        Maps (k_idx, buf_idx) → element count for each buffer involved
        in a copy (used to know when the copy counter overflows).
    """

    def __init__(self, kernels, connections, buf_depths):
        self.kernels = kernels
        self.connections = connections  # [(src_k, src_buf, dst_k, dst_buf)]
        self.buf_depths = buf_depths    # {(k_idx, buf_idx): depth}

        self.start = Signal(name="top_start")
        self.done = Signal(name="top_done")

        # Buffers driven by the copy FSM (the destination side of each conn)
        internal_bufs = {(dst_k, dst_buf) for _, _, dst_k, dst_buf in connections}

        # External write ports: input buffers NOT driven by the copy FSM
        self.ext_write_ports = {}
        for k_idx, kernel in enumerate(kernels):
            for info in kernel.buf_infos:
                buf_idx = info["idx"]
                if buf_idx == 0:
                    continue  # output buffer – written by the kernel's own FSM
                if (k_idx, buf_idx) in internal_bufs:
                    continue  # driven by copy FSM
                kw = kernel.buf_write_ports[buf_idx]
                self.ext_write_ports[(k_idx, buf_idx)] = {
                    "wen":   Signal(name=f"k{k_idx}_b{buf_idx}_ext_wen"),
                    "waddr": Signal(kw["waddr"].shape(),
                                   name=f"k{k_idx}_b{buf_idx}_ext_waddr"),
                    "wdata": Signal(kw["wdata"].shape(),
                                   name=f"k{k_idx}_b{buf_idx}_ext_wdata"),
                }

        # Output read port – wired to the last kernel's buf_read_ports[0]
        last_rp = kernels[-1].buf_read_ports[0]
        self.output_rport = {
            "raddr": Signal(last_rp["raddr"].shape(), name="top_out_raddr"),
            "rdata": Signal(last_rp["rdata"].shape(), name="top_out_rdata"),
        }

    # ------------------------------------------------------------------
    # Elaboration
    # ------------------------------------------------------------------

    def elaborate(self, platform):
        m = Module()

        # ---- Add kernel submodules ----
        for i, kernel in enumerate(self.kernels):
            m.submodules[f"k{i}"] = kernel

        # ---- Wire external ports → kernel write ports (always active) ----
        for (k_idx, buf_idx), ports in self.ext_write_ports.items():
            kernel = self.kernels[k_idx]
            wp = kernel.buf_write_ports[buf_idx]
            m.d.comb += [
                wp["wen"].eq(ports["wen"]),
                wp["waddr"].eq(ports["waddr"]),
                wp["wdata"].eq(ports["wdata"]),
            ]

        # ---- Wire output read port → last kernel's read port ----
        last_rp = self.kernels[-1].buf_read_ports[0]
        m.d.comb += [
            last_rp["raddr"].eq(self.output_rport["raddr"]),
            self.output_rport["rdata"].eq(last_rp["rdata"]),
        ]

        # ---- Copy counter (sized for largest buffer in any connection) ----
        max_depth = max(self.buf_depths.values(), default=1)
        copy_ctr = Signal(range(max_depth + 1), name="copy_ctr")

        # ---- FSM ----
        with m.FSM(init="IDLE"):
            with m.State("IDLE"):
                m.d.sync += self.done.eq(0)
                with m.If(self.start):
                    m.next = "K0_RUN"

            for i, kernel in enumerate(self.kernels):
                # K{i}_RUN: pulse start for one cycle, then wait
                with m.State(f"K{i}_RUN"):
                    m.d.comb += kernel.start.eq(1)
                    m.next = f"K{i}_WAIT"

                # K{i}_WAIT: poll done
                with m.State(f"K{i}_WAIT"):
                    with m.If(kernel.done):
                        if i < len(self.kernels) - 1:
                            m.d.sync += copy_ctr.eq(0)
                            m.next = f"COPY_{i}_{i+1}"
                        else:
                            m.d.sync += self.done.eq(1)
                            m.next = "IDLE"

                # COPY_{i}_{i+1}: DMA from kernel i output → kernel i+1 input
                if i < len(self.kernels) - 1:
                    conns_here = [
                        (s_b, d_b)
                        for (s_k, s_b, d_k, d_b) in self.connections
                        if s_k == i and d_k == i + 1
                    ]

                    with m.State(f"COPY_{i}_{i+1}"):
                        for src_buf, dst_buf in conns_here:
                            src_rp = self.kernels[i].buf_read_ports[src_buf]
                            dst_wp = self.kernels[i + 1].buf_write_ports[dst_buf]
                            m.d.comb += [
                                src_rp["raddr"].eq(copy_ctr),
                                dst_wp["wen"].eq(1),
                                dst_wp["waddr"].eq(copy_ctr),
                                dst_wp["wdata"].eq(src_rp["rdata"]),
                            ]

                        # Advance counter; use depth of first src buffer
                        if conns_here:
                            depth = self.buf_depths.get((i, conns_here[0][0]), 1)
                        else:
                            depth = 1

                        with m.If(copy_ctr == depth - 1):
                            m.d.sync += copy_ctr.eq(0)
                            m.next = f"K{i+1}_RUN"
                        with m.Else():
                            m.d.sync += copy_ctr.eq(copy_ctr + 1)

        return m


# ---------------------------------------------------------------------------
# simulate_top — run the TopModule on Amaranth simulator
# ---------------------------------------------------------------------------

def simulate_top(top, input_data, clock_period=1e-8):
    """Simulate a TopModule end-to-end.

    Parameters
    ----------
    top : TopModule
        The assembled top module.
    input_data : dict[tuple[int,int], np.ndarray]
        Maps (k_idx, buf_idx) → numpy array for each external input buffer.
    clock_period : float
        Simulation clock period in seconds.

    Returns
    -------
    output : np.ndarray
        Contents of the final kernel's output buffer (int32).
    total_cycles : int
        Clock cycles from start-pulse to done-pulse.
    wall_s : float
        Wall-clock seconds for the simulation.
    """
    sim = Simulator(top)
    sim.add_clock(clock_period)

    # Determine output buffer depth from the last kernel
    last_kernel = top.kernels[-1]
    out_info = next(i for i in last_kernel.buf_infos if i["idx"] == 0)
    out_depth = out_info["depth"]

    results = {}
    cycle_count = [0]

    async def testbench(ctx):
        # Load all external input buffers
        for (k_idx, buf_idx), data in input_data.items():
            ports = top.ext_write_ports.get((k_idx, buf_idx))
            if ports is None:
                continue
            flat = data.flatten()
            for j in range(len(flat)):
                ctx.set(ports["wen"], 1)
                ctx.set(ports["waddr"], j)
                ctx.set(ports["wdata"], int(flat[j]))
                await ctx.tick()
            ctx.set(ports["wen"], 0)
        await ctx.tick()

        # Pulse start
        ctx.set(top.start, 1)
        await ctx.tick()
        ctx.set(top.start, 0)

        # Wait for done; upper bound = sum of all kernel cycle budgets
        max_cycles = sum(
            max(info["depth"] for info in k.buf_infos) ** 2 + 100
            for k in top.kernels
        )

        for _ in range(int(max_cycles)):
            await ctx.tick()
            cycle_count[0] += 1
            if ctx.get(top.done):
                break

        # Read output
        rp = top.output_rport
        for j in range(out_depth):
            ctx.set(rp["raddr"], j)
            await ctx.tick()
            results[j] = ctx.get(rp["rdata"])

    sim.add_testbench(testbench)

    t0 = time.perf_counter()
    sim.run()
    wall = time.perf_counter() - t0

    output = np.array([results.get(i, 0) for i in range(out_depth)], dtype=np.int32)
    return output, cycle_count[0], wall
