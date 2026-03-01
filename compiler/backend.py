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
from .top_module import TopModule


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
    """Return (bit_width, is_signed) for a tinygrad base dtype.

    For floating-point types the storage is treated as an unsigned integer
    bit-vector of the appropriate width (IEEE 754 bit pattern).  Arithmetic
    on float signals in simulation will not be IEEE-accurate; use the software
    float path in the benchmark harness for numerically correct float results.
    """
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
    # Float types: stored as unsigned bit patterns
    if dtype in (dtypes.float16, dtypes.half):
        return 16, False
    if dtype in (dtypes.float32, dtypes.float):
        return 32, False
    if dtype in (dtypes.bfloat16,):
        return 16, False
    return 32, False  # fallback for unknown types


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
            # Float arrays must be loaded as IEEE 754 bit patterns, not
            # cast to int (which would give the truncated integer value).
            if np.issubdtype(data_flat.dtype, np.floating):
                nbytes = data_flat.dtype.itemsize
                uint_dtype = np.uint32 if nbytes == 4 else np.uint16
                bits_flat = data_flat.view(uint_dtype)
            else:
                bits_flat = data_flat
            for j in range(len(bits_flat)):
                ctx.set(wp["wen"], 1)
                ctx.set(wp["waddr"], j)
                ctx.set(wp["wdata"], int(bits_flat[j]))
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

    # Assemble output.  ctx.get() returns unsigned integer bit-patterns.
    # Use uint32→int32 view so: (a) int32 values get correct sign extension,
    # (b) float32 bit-patterns can be recovered via .astype(uint32).view(float32).
    raw = [results.get(i, 0) & 0xFFFFFFFF for i in range(out_depth)]
    output = np.array(raw, dtype=np.uint32).view(np.int32)
    return output, cycle_count, wall


# ---------------------------------------------------------------------------
# simulate_model — end-to-end model simulation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# compile_top_module — auto-detect kernel connections and build TopModule
# ---------------------------------------------------------------------------

def compile_top_module(schedule):
    """Compile a tinygrad schedule into a TopModule.

    Detects inter-kernel buffer connections automatically by checking Buffer
    object identity: if ``si_prev.bufs[0]`` is the same Python object as
    ``si_curr.bufs[j]`` for j ≥ 1, the two kernels are connected.

    Parameters
    ----------
    schedule : list[ExecItem]
        From Tensor.schedule().

    Returns
    -------
    top : TopModule
        Assembled top module with all kernels and copy FSM.
    connections : list[tuple[int,int,int,int]]
        Detected connections as (src_k, src_buf, dst_k, dst_buf).
    kernel_specs : list[KernelSpec]
        One KernelSpec per compute kernel (same order as in TopModule).
    """
    kernel_specs = compile_model(schedule)
    compute_items = [si for si in schedule if si.ast.op == Ops.SINK]

    # Map output buffer id → kernel index
    output_buf_ids = {}
    for k_idx, si in enumerate(compute_items):
        if si.bufs:
            output_buf_ids[id(si.bufs[0])] = k_idx

    # Detect connections: input buffer of later kernel == output buffer of earlier kernel
    connections = []
    for k_idx, si in enumerate(compute_items):
        for buf_pos, buf in enumerate(si.bufs[1:], start=1):
            if buf is not None and id(buf) in output_buf_ids:
                src_k = output_buf_ids[id(buf)]
                if src_k < k_idx:  # only forward connections
                    connections.append((src_k, 0, k_idx, buf_pos))

    # Build buf_depths: depth of each buffer involved in a copy
    buf_depths = {}
    for ks in kernel_specs:
        for info in ks.buf_infos:
            # BufferInfo dataclass with .idx and .depth
            k_idx_for_spec = kernel_specs.index(ks)
            buf_depths[(k_idx_for_spec, info.idx)] = info.depth

    kernels = [ks.kernel for ks in kernel_specs]
    top = TopModule(kernels, connections, buf_depths)
    return top, connections, kernel_specs


# ---------------------------------------------------------------------------
# count_cycles_from_schedule — analytical cycle estimator (no simulation)
# ---------------------------------------------------------------------------

def _count_cycles_from_root(root):
    """Count FSM cycles from a LoopLevel tree (no simulation required).

    Matches the cycle model of CompiledKernel's FSM:
      - Scalar kernel:          2  (IDLE → SCALAR → IDLE)
      - Single-level loop:      bound + 1
      - Two-level loop (GEMV):  outer × (inner + 2) + 1
    """
    levels = []
    level = root.body
    while level is not None:
        levels.append(level)
        level = level.body

    if not levels:
        return 2  # scalar kernel

    if len(levels) == 1:
        return levels[0].bound + 1

    # Two levels: M × (K + 2) + 1
    outer, inner = levels[0], levels[1]
    return outer.bound * (inner.bound + 2) + 1


def count_cycles_from_schedule(schedule):
    """Analytically count total FSM cycles for all kernels in *schedule*.

    Returns the sum of per-kernel cycle counts using the same model as the
    Amaranth FSM without running any simulation.  Useful for float models
    where Amaranth simulation is not bit-accurate.

    Parameters
    ----------
    schedule : list[ExecItem]
        From Tensor.schedule().

    Returns
    -------
    int
        Total cycle count across all compute kernels.
    """
    from .uop_to_ir import uop_to_ir
    from .ir import BufferMeta, DType

    renderer = HDLRenderer()
    total = 0
    for si in schedule:
        if si.ast.op != Ops.SINK:
            continue
        uops = _get_uops(si.ast, renderer)
        buf_metas = [
            BufferMeta(
                idx=b.idx,
                depth=b.depth,
                dtype=DType.from_width(b.elem_width, b.is_signed),
                is_output=b.is_output,
            )
            for b in analyze_buffers(uops)
        ]
        kernel_ir = uop_to_ir(uops, buf_metas)
        total += _count_cycles_from_root(kernel_ir.loop_tree)
    return total
