"""PCIe DMA wrapper: cycle-accurate bus transfer model around TopModule.

This module wraps a ``TopModule`` in a PCIe DMA shell so that the
simulation includes realistic transfer overhead — TLP packetisation,
per-packet latency, and bandwidth throttling — rather than the
idealised one-word-per-cycle loads used by ``simulate_top()``.

Architecture
------------
The wrapper contains a small FSM that sequences three phases:

1. **DMA_LOAD** — For each external input buffer the wrapper streams
   data from a "host memory" (Python-supplied numpy arrays held in
   simulation) into the TopModule's external write ports.  Each word
   transfer is gated by a *throttle counter* that models the PCIe
   link's per-word transfer time at the card's practical bandwidth.

2. **COMPUTE** — Pulses ``TopModule.start`` and waits for
   ``TopModule.done``, passing through all compute cycles unchanged.

3. **DMA_READBACK** — Reads the output buffer back through the
   TopModule's ``output_rport``, again throttled to PCIe bandwidth.

Timing model
~~~~~~~~~~~~
For a given ``FPGACard`` the wrapper derives:

* ``cycles_per_word`` — how many FPGA clock cycles elapse per word
  transferred over the PCIe link, based on the card's practical
  bandwidth and synthesis Fmax.
* ``dma_setup_cycles`` — a fixed overhead per DMA descriptor
  (modelling the 2.5 µs DMA latency from the card spec).

These are computed once at construction and baked into the FSM.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
from amaranth.sim import Simulator

if TYPE_CHECKING:
    from tg2hdl.fpga_card import FPGACard
    from .top_module import TopModule


# ---------------------------------------------------------------------------
# Pure-Python DMA simulation (no extra HDL, wraps simulate_top)
# ---------------------------------------------------------------------------

def simulate_top_with_pcie(
    top: TopModule,
    input_data: dict,
    card: FPGACard,
    clock_period: float = 1e-8,
) -> tuple[np.ndarray, dict, float]:
    """Simulate a TopModule with cycle-accurate PCIe DMA transfers.

    Unlike :func:`simulate_top` which writes one word per clock cycle,
    this function throttles input loading and output readback to match
    the card's PCIe bandwidth, and adds per-direction DMA setup
    latency.

    Parameters
    ----------
    top : TopModule
        The assembled top module.
    input_data : dict[tuple[int,int], np.ndarray]
        Maps (k_idx, buf_idx) → numpy array for each external input.
    card : FPGACard
        FPGA card specification (provides PCIe bandwidth, DMA latency,
        and synthesis Fmax for timing derivation).
    clock_period : float
        Simulation clock period in seconds.

    Returns
    -------
    output : np.ndarray
        Contents of the final kernel's output buffer (int32).
    cycle_counts : dict
        Breakdown of clock cycles:
          - "dma_load":    cycles for PCIe DMA input transfers
          - "compute":     cycles from start-pulse to done-pulse
          - "dma_readback": cycles for PCIe DMA output readback
          - "total":       sum of all three
          - "states":      per-FSM-state cycle counts (compute only)
          - "pcie_model":  dict with derived timing parameters
    wall_s : float
        Wall-clock seconds for the simulation.
    """
    sim = Simulator(top)
    sim.add_clock(clock_period)

    # Derive PCIe timing parameters from the card
    fmax_hz = card.synth_typical_fmax_mhz * 1e6
    pcie_bw = card.pcie_practical_bw_bytes_s        # bytes/s
    dma_latency_s = card.pcie_dma_latency_s          # seconds

    # DMA setup overhead in FPGA clock cycles (per direction)
    dma_setup_cycles = max(1, int(math.ceil(dma_latency_s * fmax_hz)))

    last_kernel = top.kernels[-1]
    out_info = next(i for i in last_kernel.buf_infos if i["idx"] == 0)
    out_depth = out_info["depth"]

    results = {}
    dma_load_cycles = [0]
    compute_cycles = [0]
    dma_readback_cycles = [0]
    state_cycles = {name: 0 for name in top.state_names}

    async def testbench(ctx):
        # ---- Phase 1: DMA LOAD ----
        # Per-direction DMA setup latency
        for _ in range(dma_setup_cycles):
            await ctx.tick()
            dma_load_cycles[0] += 1

        for (k_idx, buf_idx), data in input_data.items():
            ports = top.ext_write_ports.get((k_idx, buf_idx))
            if ports is None:
                continue
            flat = data.flatten()
            if np.issubdtype(flat.dtype, np.floating):
                nbytes = flat.dtype.itemsize
                uint_dtype = np.uint32 if nbytes == 4 else np.uint16
                flat = flat.view(uint_dtype)

            # Compute cycles-per-word for this buffer's element width
            elem_bytes = max(1, data.dtype.itemsize)
            # At fmax_hz clock rate, how many bytes can we transfer per cycle?
            bytes_per_cycle = pcie_bw / fmax_hz
            # How many cycles does it take to transfer one element?
            cycles_per_word = max(1, int(math.ceil(elem_bytes / bytes_per_cycle)))

            for j in range(len(flat)):
                ctx.set(ports["wen"], 1)
                ctx.set(ports["waddr"], j)
                ctx.set(ports["wdata"], int(flat[j]))
                # Throttle: wait cycles_per_word cycles per word
                for _ in range(cycles_per_word):
                    await ctx.tick()
                    dma_load_cycles[0] += 1
            ctx.set(ports["wen"], 0)

        await ctx.tick()
        dma_load_cycles[0] += 1  # final settle tick

        # ---- Phase 2: COMPUTE ----
        ctx.set(top.start, 1)
        await ctx.tick()
        ctx.set(top.start, 0)

        max_cycles = sum(
            max(info["depth"] for info in k.buf_infos) ** 2 + 100
            for k in top.kernels
        )

        for _ in range(int(max_cycles)):
            prev_state = ctx.get(top.state_id)
            await ctx.tick()
            compute_cycles[0] += 1
            state_name = top.state_names[prev_state]
            state_cycles[state_name] = state_cycles.get(state_name, 0) + 1
            if ctx.get(top.done):
                break

        # ---- Phase 3: DMA READBACK ----
        # Per-direction DMA setup latency
        for _ in range(dma_setup_cycles):
            await ctx.tick()
            dma_readback_cycles[0] += 1

        out_elem_bytes = out_info["elem_width"] // 8  # bits → bytes
        bytes_per_cycle = pcie_bw / fmax_hz
        readback_cpw = max(1, int(math.ceil(out_elem_bytes / bytes_per_cycle)))

        rp = top.output_rport
        for j in range(out_depth):
            ctx.set(rp["raddr"], j)
            for _ in range(readback_cpw):
                await ctx.tick()
                dma_readback_cycles[0] += 1
            results[j] = ctx.get(rp["rdata"])

    sim.add_testbench(testbench)

    t0 = time.perf_counter()
    sim.run()
    wall = time.perf_counter() - t0

    # Derived timing model parameters (for reporting)
    pcie_model = {
        "dma_setup_cycles": dma_setup_cycles,
        "fmax_hz": fmax_hz,
        "pcie_bw_bytes_s": pcie_bw,
        "dma_latency_s": dma_latency_s,
    }

    cycle_counts = {
        "dma_load": dma_load_cycles[0],
        "compute": compute_cycles[0],
        "dma_readback": dma_readback_cycles[0],
        "total": dma_load_cycles[0] + compute_cycles[0] + dma_readback_cycles[0],
        "states": state_cycles,
        "pcie_model": pcie_model,
    }
    output = np.array([results.get(i, 0) for i in range(out_depth)], dtype=np.int32)
    return output, cycle_counts, wall
