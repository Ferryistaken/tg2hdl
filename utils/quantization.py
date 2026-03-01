"""Quantization helpers for model preparation.

Quantization is a *user-level* choice — the HDL compiler is dtype-agnostic
and accepts any dtype the user expresses in the tinygrad graph.  These
utilities help convert float models into integer representations that map
naturally to FPGA arithmetic.

Workflow
--------
1. Train or load a float32 model.
2. Quantize weights/activations with ``quantize_int8`` / ``quantize_int16``.
3. Express the quantized model in tinygrad using ``dtypes.int8`` / ``dtypes.int32``.
4. Compile and simulate via the HDL compiler.
5. Use ``dequantize`` on the output to recover float-scale predictions.

Scale convention
----------------
All functions use *symmetric per-tensor* quantization:

    float_val ≈ int_val × scale
    int_val  = round(float_val / scale).clip(-MAX, MAX)

where ``MAX = 2^(n_bits-1) - 1`` (127 for int8, 32767 for int16).
"""

import struct
import numpy as np


# ---------------------------------------------------------------------------
# Integer quantization
# ---------------------------------------------------------------------------

def quantize_int8(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor INT8 quantization.

    Parameters
    ----------
    arr : np.ndarray
        Input array (any float dtype).

    Returns
    -------
    quantized : np.ndarray, dtype=np.int8
    scale : float
        Multiply by this to approximately recover the original values.
    """
    arr = np.asarray(arr, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        return np.zeros_like(arr, dtype=np.int8), 1.0
    scale = max_abs / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, scale


def quantize_int16(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor INT16 quantization.

    Parameters
    ----------
    arr : np.ndarray
        Input array (any float dtype).

    Returns
    -------
    quantized : np.ndarray, dtype=np.int16
    scale : float
    """
    arr = np.asarray(arr, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        return np.zeros_like(arr, dtype=np.int16), 1.0
    scale = max_abs / 32767.0
    q = np.clip(np.round(arr / scale), -32767, 32767).astype(np.int16)
    return q, scale


def dequantize(arr: np.ndarray, scale: float) -> np.ndarray:
    """Convert a quantized integer array back to float32.

    Parameters
    ----------
    arr : np.ndarray
        Integer array (int8 or int16 or int32).
    scale : float
        The scale factor from the quantize step.

    Returns
    -------
    np.ndarray, dtype=float32
    """
    return arr.astype(np.float32) * scale


# ---------------------------------------------------------------------------
# IEEE 754 bit-pattern conversion (for float16/float32 HDL simulation)
# ---------------------------------------------------------------------------

def float_to_bits(arr: np.ndarray) -> np.ndarray:
    """Reinterpret IEEE 754 floats as unsigned integer bit patterns.

    float16  → uint16
    float32  → uint32
    float64  → uint64  (use sparingly — HDL rarely has 64-bit float paths)

    This is the representation used when loading float data into HDL
    memory-mapped buffers that store float values as bit vectors.

    Parameters
    ----------
    arr : np.ndarray
        Array with a floating-point dtype.

    Returns
    -------
    np.ndarray
        Same shape, dtype is the corresponding unsigned integer type.
    """
    arr = np.asarray(arr)
    if arr.dtype == np.float16:
        return arr.view(np.uint16)
    if arr.dtype == np.float32:
        return arr.view(np.uint32)
    if arr.dtype == np.float64:
        return arr.view(np.uint64)
    raise TypeError(f"float_to_bits: unsupported dtype {arr.dtype}")


def bits_to_float(arr: np.ndarray, float_dtype=np.float32) -> np.ndarray:
    """Reinterpret unsigned integer bit patterns as IEEE 754 floats.

    Inverse of ``float_to_bits``.

    Parameters
    ----------
    arr : np.ndarray
        Integer array (uint16, uint32, or uint64).
    float_dtype : dtype
        Target float type (default float32).

    Returns
    -------
    np.ndarray
        Same shape, reinterpreted as the given float dtype.
    """
    arr = np.asarray(arr)
    target = np.dtype(float_dtype)
    uint_map = {np.float16: np.uint16, np.float32: np.uint32, np.float64: np.uint64}
    expected_uint = uint_map.get(target.type)
    if expected_uint is None:
        raise TypeError(f"bits_to_float: unsupported target dtype {float_dtype}")
    return arr.astype(expected_uint).view(target)
