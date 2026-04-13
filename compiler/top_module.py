"""TopModule: sequences compiled kernels with dependency-driven copy states.

TopModule wires multiple CompiledKernel instances into a single Amaranth
Elaboratable. A copy FSM transfers a producer kernel's output buffer into
the input buffers of whichever later kernels depend on it.

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

FSM sequence (example)
----------------------
IDLE → K0_RUN → K0_WAIT → COPY_K0_G0 → K1_RUN → K1_WAIT → K2_RUN → ...
"""

import time

import numpy as np
from amaranth.hdl import Elaboratable, Module, Signal
from amaranth.sim import Simulator


class TopModule(Elaboratable):
    """Hardware sequencer for N compiled kernels.

    Parameters
    ----------
    kernels : list[CompiledKernel]
        Kernels to run in order.
    connections : list[tuple[int, int, int, int]]
        Each entry is (src_k, src_buf, dst_k, dst_buf): the source buffer of
        kernel ``src_k`` is DMA-copied into input buffer ``dst_buf`` of kernel
        ``dst_k`` after ``src_k`` completes.
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

        self._copy_groups = self._build_copy_groups()
        self.state_names = self._build_state_names()
        self.state_id = Signal(range(max(len(self.state_names), 1)), name="top_state_id")

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

    def _build_copy_groups(self):
        """Group connections by source kernel and source buffer.

        Each group can be copied in one FSM state because all destinations are
        driven from the same source read port address.
        """
        groups_by_src = {k_idx: [] for k_idx in range(len(self.kernels))}
        grouped = {}
        for src_k, src_buf, dst_k, dst_buf in self.connections:
            key = (src_k, src_buf)
            grouped.setdefault(key, []).append((src_k, src_buf, dst_k, dst_buf))

        for (src_k, _src_buf), conns in grouped.items():
            groups_by_src[src_k].append(conns)

        for src_k in groups_by_src:
            groups_by_src[src_k].sort(key=lambda grp: (grp[0][1], grp[0][2], grp[0][3]))
        return groups_by_src

    def _build_state_names(self):
        names = ["IDLE"]
        for i in range(len(self.kernels)):
            names.append(f"K{i}_RUN")
            names.append(f"K{i}_WAIT")
            for gi, _group in enumerate(self._copy_groups.get(i, [])):
                names.append(f"K{i}_COPY_{gi}")
        return names

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

        def next_state_after_kernel(i: int) -> str:
            groups = self._copy_groups.get(i, [])
            if groups:
                return f"K{i}_COPY_0"
            if i < len(self.kernels) - 1:
                return f"K{i+1}_RUN"
            return "IDLE"

        # ---- FSM ----
        with m.FSM(init="IDLE"):
            with m.State("IDLE"):
                m.d.comb += self.state_id.eq(self.state_names.index("IDLE"))
                m.d.sync += self.done.eq(0)
                with m.If(self.start):
                    m.next = "K0_RUN"

            for i, kernel in enumerate(self.kernels):
                # K{i}_RUN: pulse start for one cycle, then wait
                with m.State(f"K{i}_RUN"):
                    m.d.comb += self.state_id.eq(self.state_names.index(f"K{i}_RUN"))
                    m.d.comb += kernel.start.eq(1)
                    m.next = f"K{i}_WAIT"

                # K{i}_WAIT: poll done
                with m.State(f"K{i}_WAIT"):
                    m.d.comb += self.state_id.eq(self.state_names.index(f"K{i}_WAIT"))
                    with m.If(kernel.done):
                        if i == len(self.kernels) - 1 and not self._copy_groups.get(i):
                            m.d.sync += self.done.eq(1)
                            m.next = "IDLE"
                        else:
                            m.d.sync += copy_ctr.eq(0)
                            m.next = next_state_after_kernel(i)

                for gi, group in enumerate(self._copy_groups.get(i, [])):
                    state_name = f"K{i}_COPY_{gi}"
                    next_group = gi + 1
                    if next_group < len(self._copy_groups.get(i, [])):
                        next_state = f"K{i}_COPY_{next_group}"
                    elif i < len(self.kernels) - 1:
                        next_state = f"K{i+1}_RUN"
                    else:
                        next_state = "IDLE"

                    with m.State(state_name):
                        m.d.comb += self.state_id.eq(self.state_names.index(state_name))
                        src_k, src_buf, _, _ = group[0]
                        src_rp = self.kernels[src_k].buf_read_ports[src_buf]
                        m.d.comb += src_rp["raddr"].eq(copy_ctr)

                        for _s_k, _s_buf, dst_k, dst_buf in group:
                            dst_wp = self.kernels[dst_k].buf_write_ports[dst_buf]
                            m.d.comb += [
                                dst_wp["wen"].eq(1),
                                dst_wp["waddr"].eq(copy_ctr),
                                dst_wp["wdata"].eq(src_rp["rdata"]),
                            ]

                        depth = self.buf_depths.get((src_k, src_buf), 1)
                        with m.If(copy_ctr == depth - 1):
                            m.d.sync += copy_ctr.eq(0)
                            if next_state == "IDLE":
                                m.d.sync += self.done.eq(1)
                            m.next = next_state
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
    cycle_counts : dict
        Breakdown of clock cycles:
          - "load":    cycles spent writing input data into BRAM
          - "compute": cycles from start-pulse to done-pulse
          - "readback": cycles spent reading output data from BRAM
          - "total":   sum of all three
          - "states":  cycles spent in each top-level FSM state during compute
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
    load_cycles = [0]
    compute_cycles = [0]
    readback_cycles = [0]
    state_cycles = {name: 0 for name in top.state_names}

    async def testbench(ctx):
        # Load all external input buffers
        for (k_idx, buf_idx), data in input_data.items():
            ports = top.ext_write_ports.get((k_idx, buf_idx))
            if ports is None:
                continue
            flat = data.flatten()
            if np.issubdtype(flat.dtype, np.floating):
                nbytes = flat.dtype.itemsize
                uint_dtype = np.uint32 if nbytes == 4 else np.uint16
                flat = flat.view(uint_dtype)
            for j in range(len(flat)):
                ctx.set(ports["wen"], 1)
                ctx.set(ports["waddr"], j)
                ctx.set(ports["wdata"], int(flat[j]))
                await ctx.tick()
                load_cycles[0] += 1
            ctx.set(ports["wen"], 0)
        await ctx.tick()
        load_cycles[0] += 1  # final settle tick

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
            compute_cycles[0] += 1
            state_name = top.state_names[ctx.get(top.state_id)]
            state_cycles[state_name] = state_cycles.get(state_name, 0) + 1
            if ctx.get(top.done):
                break

        # Read output
        rp = top.output_rport
        for j in range(out_depth):
            ctx.set(rp["raddr"], j)
            await ctx.tick()
            readback_cycles[0] += 1
            results[j] = ctx.get(rp["rdata"])

    sim.add_testbench(testbench)

    t0 = time.perf_counter()
    sim.run()
    wall = time.perf_counter() - t0

    cycle_counts = {
        "load": load_cycles[0],
        "compute": compute_cycles[0],
        "readback": readback_cycles[0],
        "total": load_cycles[0] + compute_cycles[0] + readback_cycles[0],
        "states": state_cycles,
    }
    output = np.array([results.get(i, 0) for i in range(out_depth)], dtype=np.int32)
    return output, cycle_counts, wall
