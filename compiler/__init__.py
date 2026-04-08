"""tinygrad UOps → Amaranth HDL compiler.

Public API:
    HDLRenderer             — tinygrad Renderer for sequential hardware
    uops_to_kernel_ir       — list[UOp] → (KernelIR, buf_infos)
    compile_kernel          — list[UOp] → KernelIR → CompiledKernel (Amaranth Elaboratable)
    compile_model           — tinygrad schedule → list[KernelSpec]
    compile_top_module      — tinygrad schedule → TopModule (auto-connects kernels)
    simulate_kernel         — run a CompiledKernel on Amaranth simulator
    simulate_top            — run a TopModule on Amaranth simulator
    count_cycles_from_schedule — analytical cycle estimator (no simulation)

Quantization utilities are in the ``utils`` package, not here:
    from utils import quantize_int8, quantize_int16, dequantize
"""

from .backend import (
    HDLRenderer,
    uops_to_kernel_ir,
    compile_kernel,
    compile_model,
    compile_top_module,
    simulate_kernel,
    count_cycles_from_schedule,
)
from .hdl_module import CompiledKernel
from .top_module import TopModule, simulate_top

from .transforms import unroll_loop, unroll_reduce
from .utils import pretty_print_uops, synthesis_stats, show_hardware
from .visualize import analyze_schedule, analyze_tensor, PipelineView, KernelView
