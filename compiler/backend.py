"""HDL backend for tinygrad: UOps → Amaranth hardware.

Provides HDLRenderer (tells tinygrad we're a sequential device),
compile_kernel (UOps → CompiledKernel), compile_model (schedule → top module),
and simulate (run on Amaranth simulator).
"""

import os
import time
from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.renderer import Renderer
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.uop.ops import Ops, KernelInfo, AxisType
from tinygrad.dtype import AddrSpace

from amaranth.sim import Simulator

from .hdl_module import CompiledKernel


# ---------------------------------------------------------------------------
# HDLRenderer — tinygrad Renderer subclass
# ---------------------------------------------------------------------------

class HDLRenderer(Renderer):
    """Tells tinygrad to generate sequential UOps (no GPU features)."""

    device = "HDL"
    has_local = False
    has_shared = False
    supports_float4 = False
    global_max = None
    local_max = None

    def render(self, uops):
        # We don't use rendered source code — we consume UOps directly.
        return "// HDL backend — UOps consumed directly"


# ---------------------------------------------------------------------------
# Buffer / kernel analysis
# ---------------------------------------------------------------------------

@dataclass
class BufferInfo:
    """Descriptor for a DEFINE_GLOBAL buffer."""
    idx: int          # buffer index (arg of DEFINE_GLOBAL)
    depth: int        # number of elements
    elem_width: int   # bits per element (8 for int8, 32 for int32)
    is_signed: bool
    is_output: bool   # True if this is the output buffer (idx == 0)


def _get_uops(ast, renderer):
    """Linearize an AST UOp into a list of sequential UOps."""
    if ast.arg is None:
        ast = ast.replace(arg=KernelInfo())
    sink = full_rewrite_to_sink(ast, renderer)
    return linearize(sink)


def _dtype_info(dtype):
    """Return (bit_width, is_signed) for a tinygrad base dtype."""
    if dtype in (dtypes.char, dtypes.int8):
        return 8, True
    if dtype in (dtypes.uchar, dtypes.uint8):
        return 8, False
    if dtype in (dtypes.short, dtypes.int16):
        return 16, True
    if dtype in (dtypes.int, dtypes.int32):
        return 32, True
    if dtype in (dtypes.uint, dtypes.uint32):
        return 32, False
    return 32, True  # fallback


def analyze_buffers(uops):
    """Extract BufferInfo for each DEFINE_GLOBAL in the UOp list."""
    buffers = []
    for u in uops:
        if u.op == Ops.DEFINE_GLOBAL:
            depth = u.dtype.size if u.dtype.size > 0 else 1
            w, s = _dtype_info(u.dtype.base)
            buffers.append(BufferInfo(
                idx=u.arg,
                depth=depth,
                elem_width=w,
                is_signed=s,
                is_output=(u.arg == 0),
            ))
    return buffers


# ---------------------------------------------------------------------------
# compile_kernel
# ---------------------------------------------------------------------------

def compile_kernel(uops):
    """Compile a linearized UOp list into a CompiledKernel (Amaranth Elaboratable).

    Parameters
    ----------
    uops : list[UOp]
        Linearized UOps from tinygrad (via _get_uops).

    Returns
    -------
    CompiledKernel
        Amaranth Elaboratable ready for simulation or synthesis.
    """
    buf_infos_raw = analyze_buffers(uops)
    buf_infos = [
        {
            "idx": b.idx,
            "depth": b.depth,
            "elem_width": b.elem_width,
            "is_signed": b.is_signed,
            "is_output": b.is_output,
        }
        for b in buf_infos_raw
    ]
    return CompiledKernel(uops, buf_infos)


# ---------------------------------------------------------------------------
# compile_model
# ---------------------------------------------------------------------------

@dataclass
class KernelSpec:
    """Info about a compiled kernel and its buffer mapping."""
    kernel: CompiledKernel
    uops: list
    buf_infos: list  # list of BufferInfo
    # Maps kernel-local buffer index → (name, shape) for the model
    buf_map: dict  # buf_idx → schedule buffer reference


def compile_model(schedule):
    """Compile a tinygrad schedule into a list of KernelSpecs.

    Parameters
    ----------
    schedule : list[ExecItem]
        From Tensor.schedule().

    Returns
    -------
    list[KernelSpec]
        One KernelSpec per compute kernel in the schedule.
    """
    renderer = HDLRenderer()
    kernels = []

    for si in schedule:
        ast = si.ast
        if ast.op != Ops.SINK:
            continue
        uops = _get_uops(ast, renderer)
        buf_infos = analyze_buffers(uops)
        kernel = compile_kernel(uops)
        kernels.append(KernelSpec(
            kernel=kernel,
            uops=uops,
            buf_infos=buf_infos,
            buf_map={},
        ))

    return kernels


# ---------------------------------------------------------------------------
# simulate — run a compiled kernel on the Amaranth simulator
# ---------------------------------------------------------------------------

def simulate_kernel(kernel, input_data, clock_period=1e-8):
    """Simulate a single CompiledKernel.

    Parameters
    ----------
    kernel : CompiledKernel
        The compiled kernel module.
    input_data : dict[int, np.ndarray]
        Maps buffer index → numpy array of data to load.
        Buffer 0 (output) is not loaded.
    clock_period : float
        Simulation clock period in seconds (default 10ns = 100MHz).

    Returns
    -------
    output : np.ndarray
        Output buffer contents after computation.
    cycles : int
        Number of compute cycles.
    wall_time : float
        Wall-clock seconds for simulation.
    """
    sim = Simulator(kernel)
    sim.add_clock(clock_period)

    # Find output buffer info
    out_info = None
    for info in kernel.buf_infos:
        if info["idx"] == 0:
            out_info = info
            break
    assert out_info is not None, "No output buffer (idx=0) found"

    out_depth = out_info["depth"]
    results = {}
    cycle_count = 0

    async def testbench(ctx):
        nonlocal cycle_count

        # Load input data into buffers
        for buf_idx, data in input_data.items():
            wp = kernel.buf_write_ports[buf_idx]
            data_flat = data.flatten()
            for j in range(len(data_flat)):
                ctx.set(wp["wen"], 1)
                ctx.set(wp["waddr"], j)
                ctx.set(wp["wdata"], int(data_flat[j]))
                await ctx.tick()
            ctx.set(wp["wen"], 0)
        await ctx.tick()

        # Start computation
        ctx.set(kernel.start, 1)
        await ctx.tick()
        ctx.set(kernel.start, 0)

        # Wait for done
        # Upper bound: outer × (inner + 2) + margin
        max_cycles = 1
        for info in kernel.buf_infos:
            max_cycles = max(max_cycles, info["depth"])
        max_cycles = max_cycles * max_cycles + 100

        for _ in range(max_cycles):
            await ctx.tick()
            cycle_count += 1
            if ctx.get(kernel.done):
                break

        # Read output buffer
        rp = kernel.buf_read_ports[0]
        for j in range(out_depth):
            ctx.set(rp["raddr"], j)
            await ctx.tick()
            results[j] = ctx.get(rp["rdata"])

    sim.add_testbench(testbench)

    t0 = time.perf_counter()
    sim.run()
    wall = time.perf_counter() - t0

    # Assemble output
    output = np.array([results.get(i, 0) for i in range(out_depth)], dtype=np.int32)
    return output, cycle_count, wall


# ---------------------------------------------------------------------------
# simulate_model — end-to-end model simulation
# ---------------------------------------------------------------------------

def quantize_int8(arr):
    """Symmetric per-tensor INT8 quantization. Returns (int8_array, scale)."""
    max_abs = np.max(np.abs(arr))
    if max_abs == 0:
        return np.zeros_like(arr, dtype=np.int8), 1.0
    scale = max_abs / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, scale
