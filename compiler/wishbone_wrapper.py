"""Wishbone slave wrapper: maps TopModule onto a standard Wishbone bus.

This module wraps a ``TopModule`` in a Wishbone B4 pipelined slave so
that it can be integrated into a LiteX SoC or any Wishbone master.
It is **synthesizable** — the wrapper goes through Yosys/nextpnr
alongside the TopModule and shows up in resource utilisation reports.

Register Map
------------
All addresses are 32-bit word-addressed (each address selects 4 bytes).

==========  ====  ========================================================
Offset      R/W   Description
==========  ====  ========================================================
0x0000      R/W   **CTRL** — write ``1`` to start compute; reads back
                  ``{done, busy}`` in bits [1:0].
0x0004      R     **STATUS** — bits [1:0] = ``{done, busy}``; bits [31:2]
                  reserved.
0x0008      R     **CYCLE_CNT** — 32-bit cycle counter (resets on start,
                  stops on done).

0x1000+     W     **Input buffer region** — each external input buffer is
                  assigned a contiguous block of word addresses starting
                  at ``0x1000``.  Writing to address ``0x1000 + offset``
                  stores the data word at element ``offset`` in the first
                  external buffer, etc.  Buffers are laid out sequentially
                  in the order returned by ``TopModule.ext_write_ports``.

0x8000+     R     **Output buffer region** — reading from address
                  ``0x8000 + j`` returns element ``j`` of the final
                  kernel's output buffer.
==========  ====  ========================================================

Wishbone signals
~~~~~~~~~~~~~~~~
Classic pipelined Wishbone B4:

* ``cyc_i``, ``stb_i`` — bus cycle and strobe (active-high)
* ``we_i`` — write-enable
* ``adr_i`` — word address (top bits select region, lower bits select
  element)
* ``dat_w_i`` — write data (32-bit)
* ``dat_r_o`` — read data (32-bit)
* ``ack_o`` — acknowledge (one-cycle pulse)
* ``sel_i`` — byte-lane select (unused; always full-word)
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
from amaranth.hdl import Elaboratable, Module, Signal, Cat, Const
from amaranth.sim import Simulator

if TYPE_CHECKING:
    from tg2hdl.fpga_card import FPGACard
    from .top_module import TopModule


# -- Address region bases (word-addressed) --
_CTRL_ADDR = 0x0000
_STATUS_ADDR = 0x0004 >> 2  # word address 1
_CYCLE_CNT_ADDR = 0x0008 >> 2  # word address 2
_INPUT_BASE = 0x1000 >> 2  # word address 0x400
_OUTPUT_BASE = 0x8000 >> 2  # word address 0x2000


class WishboneTopWrapper(Elaboratable):
    """Wishbone B4 slave that wraps a TopModule.

    Parameters
    ----------
    top : TopModule
        The compiled top-level module to wrap.
    data_width : int
        Bus data width in bits (default 32).
    addr_width : int
        Bus address width in bits (default 16, gives 64 K word address
        space = 256 KB byte-addressable).
    """

    def __init__(self, top: TopModule, data_width: int = 32, addr_width: int = 16):
        self.top = top
        self.data_width = data_width
        self.addr_width = addr_width

        # -- Wishbone slave signals --
        self.cyc_i = Signal(name="wb_cyc_i")
        self.stb_i = Signal(name="wb_stb_i")
        self.we_i = Signal(name="wb_we_i")
        self.adr_i = Signal(addr_width, name="wb_adr_i")
        self.dat_w_i = Signal(data_width, name="wb_dat_w_i")
        self.dat_r_o = Signal(data_width, name="wb_dat_r_o")
        self.sel_i = Signal(data_width // 8, name="wb_sel_i")
        self.ack_o = Signal(name="wb_ack_o")

        # Build the input buffer layout: ordered list of (key, depth)
        self._input_layout = []
        offset = 0
        for key in sorted(top.ext_write_ports.keys()):
            k_idx, buf_idx = key
            # Find depth from the kernel's buf_infos
            kernel = top.kernels[k_idx]
            info = next(i for i in kernel.buf_infos if i["idx"] == buf_idx)
            depth = info["depth"]
            self._input_layout.append((key, offset, depth))
            offset += depth
        self._input_total = offset

        # Output depth
        last_kernel = top.kernels[-1]
        out_info = next(i for i in last_kernel.buf_infos if i["idx"] == 0)
        self._output_depth = out_info["depth"]

    def elaborate(self, platform):
        m = Module()
        m.submodules.top = self.top

        top = self.top

        # -- Internal state --
        busy = Signal(name="wb_busy")
        done_latched = Signal(name="wb_done_latched")
        cycle_counter = Signal(32, name="wb_cycle_counter")

        # Latch done from TopModule (sticky until next start)
        with m.If(top.done):
            m.d.sync += done_latched.eq(1)
            m.d.sync += busy.eq(0)

        # Count cycles while busy
        with m.If(busy & ~done_latched):
            m.d.sync += cycle_counter.eq(cycle_counter + 1)

        # -- Wishbone bus logic --
        # Default: no ack, no data
        m.d.sync += self.ack_o.eq(0)

        # Clear TopModule start (only pulse for one cycle)
        m.d.comb += top.start.eq(0)

        # Default: deassert all external write enables
        for key, ports in top.ext_write_ports.items():
            m.d.comb += ports["wen"].eq(0)

        with m.If(self.cyc_i & self.stb_i):
            m.d.sync += self.ack_o.eq(1)

            with m.If(self.we_i):
                # ---- WRITE ----
                with m.If(self.adr_i == _CTRL_ADDR):
                    # Write 1 to CTRL → start compute
                    with m.If(self.dat_w_i[0]):
                        m.d.comb += top.start.eq(1)
                        m.d.sync += busy.eq(1)
                        m.d.sync += done_latched.eq(0)
                        m.d.sync += cycle_counter.eq(0)

                # Input buffer region
                for key, buf_offset, depth in self._input_layout:
                    ports = top.ext_write_ports[key]
                    region_start = _INPUT_BASE + buf_offset
                    region_end = region_start + depth
                    with m.If((self.adr_i >= region_start) &
                              (self.adr_i < region_end)):
                        local_addr = Signal(
                            range(depth),
                            name=f"waddr_k{key[0]}_b{key[1]}",
                        )
                        m.d.comb += local_addr.eq(self.adr_i - region_start)
                        m.d.comb += [
                            ports["wen"].eq(1),
                            ports["waddr"].eq(local_addr),
                            ports["wdata"].eq(self.dat_w_i),
                        ]

            with m.Else():
                # ---- READ ----
                with m.If(self.adr_i == _CTRL_ADDR):
                    m.d.sync += self.dat_r_o.eq(Cat(busy, done_latched))

                with m.Elif(self.adr_i == _STATUS_ADDR):
                    m.d.sync += self.dat_r_o.eq(Cat(busy, done_latched))

                with m.Elif(self.adr_i == _CYCLE_CNT_ADDR):
                    m.d.sync += self.dat_r_o.eq(cycle_counter)

                # Output buffer region
                rp = top.output_rport
                with m.Elif((self.adr_i >= _OUTPUT_BASE) &
                            (self.adr_i < _OUTPUT_BASE + self._output_depth)):
                    local_addr = Signal(
                        range(self._output_depth),
                        name="raddr_out",
                    )
                    m.d.comb += local_addr.eq(self.adr_i - _OUTPUT_BASE)
                    m.d.comb += rp["raddr"].eq(local_addr)
                    m.d.sync += self.dat_r_o.eq(rp["rdata"])

                with m.Else():
                    m.d.sync += self.dat_r_o.eq(0)

        return m


# ---------------------------------------------------------------------------
# Simulation: drive TopModule through the Wishbone bus interface
# ---------------------------------------------------------------------------

def simulate_wishbone(
    top: TopModule,
    input_data: dict,
    card: FPGACard | None = None,
    clock_period: float = 1e-8,
) -> tuple[np.ndarray, dict, float]:
    """Simulate a TopModule through its Wishbone bus wrapper.

    This creates a :class:`WishboneTopWrapper`, instantiates the Amaranth
    simulator, and drives data through the Wishbone bus — every input
    write, control register poke, status poll, and output read is a real
    bus transaction that takes one clock cycle.

    The simulation is **cycle-accurate**: the total cycle count includes
    all bus overhead (one cycle per Wishbone transaction).

    Parameters
    ----------
    top : TopModule
        The assembled top module.
    input_data : dict[tuple[int,int], np.ndarray]
        Maps (k_idx, buf_idx) → numpy array for each external input.
    card : FPGACard or None
        FPGA card specification (used for metadata in the returned
        cycle counts, not for throttling — the bus itself is the
        bottleneck model).
    clock_period : float
        Simulation clock period in seconds.

    Returns
    -------
    output : np.ndarray
        Contents of the final kernel's output buffer (int32).
    cycle_counts : dict
        Breakdown of clock cycles:
          - "wb_load":     cycles for Wishbone input writes
          - "wb_start":    cycles for control register write
          - "compute":     cycles from start to done (polled via bus)
          - "wb_readback": cycles for Wishbone output reads
          - "wb_total":    sum of all phases
          - "states":      per-FSM-state cycle counts (compute only)
    wall_s : float
        Wall-clock seconds for the simulation.
    """
    wrapper = WishboneTopWrapper(top)
    sim = Simulator(wrapper)
    sim.add_clock(clock_period)

    last_kernel = top.kernels[-1]
    out_info = next(i for i in last_kernel.buf_infos if i["idx"] == 0)
    out_depth = out_info["depth"]

    results = {}
    load_cycles = [0]
    start_cycles = [0]
    compute_cycles = [0]
    readback_cycles = [0]
    state_cycles = {name: 0 for name in top.state_names}

    async def testbench(ctx):
        # Helper: perform a single Wishbone write transaction
        async def wb_write(addr: int, data: int):
            ctx.set(wrapper.cyc_i, 1)
            ctx.set(wrapper.stb_i, 1)
            ctx.set(wrapper.we_i, 1)
            ctx.set(wrapper.adr_i, addr)
            ctx.set(wrapper.dat_w_i, data)
            await ctx.tick()
            ctx.set(wrapper.stb_i, 0)
            ctx.set(wrapper.we_i, 0)
            await ctx.tick()  # wait for ack

        # Helper: perform a single Wishbone read transaction
        async def wb_read(addr: int) -> int:
            ctx.set(wrapper.cyc_i, 1)
            ctx.set(wrapper.stb_i, 1)
            ctx.set(wrapper.we_i, 0)
            ctx.set(wrapper.adr_i, addr)
            await ctx.tick()
            ctx.set(wrapper.stb_i, 0)
            await ctx.tick()  # wait for ack + data
            return ctx.get(wrapper.dat_r_o)

        # ---- Phase 1: Load input data via Wishbone writes ----
        for key, buf_offset, depth in wrapper._input_layout:
            data = input_data.get(key)
            if data is None:
                continue
            flat = data.flatten()
            if np.issubdtype(flat.dtype, np.floating):
                nbytes = flat.dtype.itemsize
                uint_dtype = np.uint32 if nbytes == 4 else np.uint16
                flat = flat.view(uint_dtype)
            for j in range(len(flat)):
                wb_addr = _INPUT_BASE + buf_offset + j
                await wb_write(wb_addr, int(flat[j]))
                load_cycles[0] += 2  # each wb_write takes 2 cycles

        # ---- Phase 2: Start compute via CTRL register ----
        await wb_write(_CTRL_ADDR, 1)
        start_cycles[0] += 2

        # ---- Phase 3: Poll STATUS until done ----
        max_polls = sum(
            max(info["depth"] for info in k.buf_infos) ** 2 + 100
            for k in top.kernels
        )

        for _ in range(int(max_polls)):
            prev_state = ctx.get(top.state_id)
            status = await wb_read(_STATUS_ADDR)
            compute_cycles[0] += 2  # each wb_read takes 2 cycles
            state_name = top.state_names[prev_state]
            state_cycles[state_name] = state_cycles.get(state_name, 0) + 1
            done_bit = (status >> 1) & 1
            if done_bit:
                break

        # ---- Phase 4: Read output via Wishbone reads ----
        for j in range(out_depth):
            val = await wb_read(_OUTPUT_BASE + j)
            readback_cycles[0] += 2  # each wb_read takes 2 cycles
            results[j] = val

        # Release bus
        ctx.set(wrapper.cyc_i, 0)

    sim.add_testbench(testbench)

    t0 = time.perf_counter()
    sim.run()
    wall = time.perf_counter() - t0

    total = load_cycles[0] + start_cycles[0] + compute_cycles[0] + readback_cycles[0]

    cycle_counts = {
        "wb_load": load_cycles[0],
        "wb_start": start_cycles[0],
        "compute": compute_cycles[0],
        "wb_readback": readback_cycles[0],
        "wb_total": total,
        "states": state_cycles,
    }
    output = np.array([results.get(i, 0) for i in range(out_depth)], dtype=np.int32)
    return output, cycle_counts, wall
