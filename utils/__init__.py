"""Utility helpers for the ml_kernels_hdl project.

Public API:
    quantize_int8   — symmetric per-tensor INT8 quantization
    quantize_int16  — symmetric per-tensor INT16 quantization
    dequantize      — reverse quantization (int → float)
    float_to_bits   — IEEE 754 float → uint bit pattern (np.ndarray)
    bits_to_float   — uint bit pattern → IEEE 754 float (np.ndarray)
"""

from .quantization import (
    quantize_int8,
    quantize_int16,
    dequantize,
    float_to_bits,
    bits_to_float,
)
