"""Basic tinygrad UOps → HDL compiler.

Public API:
    HDLRenderer     — tinygrad Renderer for sequential hardware
    compile_kernel  — list[UOp] → CompiledKernel (Amaranth Elaboratable)
    compile_model   — tinygrad schedule → list[KernelSpec]
    simulate_kernel — run a CompiledKernel on Amaranth simulator
    quantize_int8   — symmetric INT8 quantization helper
"""

from .backend import HDLRenderer, compile_kernel, compile_model, simulate_kernel, quantize_int8
from .hdl_module import CompiledKernel
