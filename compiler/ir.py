"""Typed Kernel IR for the tg2hdl compiler.

Defines the intermediate representation that sits between tinygrad UOps and
Amaranth RTL generation.  All dtype information is explicit and self-contained
so that downstream lowering passes do not need to import tinygrad.

Types
-----
DType         — scalar data type enum (INT8, INT32, FP32, ...)
IRConst       — compile-time constant value
IRCounter     — loop induction variable
IRBufLoad     — value read from a global buffer
IRRegLoad     — value read from the accumulator register
IROp          — arithmetic/logical operation result
IRValue       — union of the above (type alias)
IRBufStore    — write a value to a global buffer (side effect)
IRRegStore    — write a value to the accumulator register (side effect)
LoopIR        — one level of a loop nest, with prologue/epilogue stores
BufferMeta    — descriptor for a DEFINE_GLOBAL buffer
KernelIR      — top-level typed kernel representation
"""

from __future__ import annotations
from typing import Any

import enum
import struct
from dataclasses import dataclass, field
from typing import Union


# ---------------------------------------------------------------------------
# DType
# ---------------------------------------------------------------------------

class DType(enum.Enum):
    """Scalar data types supported by the compiler.

    Each member stores (name, bit_width, is_signed, is_float).
    """
    INT8   = ("int8",   8,  True,  False)
    INT16  = ("int16",  16, True,  False)
    INT32  = ("int32",  32, True,  False)
    UINT8  = ("uint8",  8,  False, False)
    UINT16 = ("uint16", 16, False, False)
    UINT32 = ("uint32", 32, False, False)
    FP16   = ("fp16",   16, False, True)
    BF16   = ("bf16",   16, False, True)
    FP32   = ("fp32",   32, False, True)

    def __init__(self, _name, bit_width: int, is_signed: bool, is_float: bool):
        self.bit_width = bit_width
        self.is_signed = is_signed
        self.is_float = is_float

    def amaranth_shape(self):
        """Return an Amaranth Shape (signed/unsigned) for this dtype."""
        from amaranth.hdl import signed, unsigned
        return signed(self.bit_width) if self.is_signed else unsigned(self.bit_width)

    @classmethod
    def from_tinygrad(cls, tg_dtype) -> "DType":
        """Map a tinygrad dtype to DType.

        Raises ValueError for unsupported dtypes (fail-loud policy).
        """
        from tinygrad import dtypes
        _MAP = {
            dtypes.int8:    cls.INT8,
            dtypes.int16:   cls.INT16,
            dtypes.int32:   cls.INT32,
            dtypes.int:     cls.INT32,
            dtypes.uint8:   cls.UINT8,
            dtypes.uint16:  cls.UINT16,
            dtypes.uint32:  cls.UINT32,
            dtypes.uint:    cls.UINT32,
            dtypes.float16: cls.FP16,
            dtypes.half:    cls.FP16,
            dtypes.bfloat16: cls.BF16,
            dtypes.float32: cls.FP32,
            dtypes.float:   cls.FP32,
        }
        if tg_dtype not in _MAP:
            raise ValueError(
                f"Unsupported dtype: {tg_dtype!r}. "
                "Add an entry to DType._MAP or use a supported type."
            )
        return _MAP[tg_dtype]

    @classmethod
    def from_width(cls, bit_width: int, is_signed: bool) -> "DType":
        """Reconstruct DType from (bit_width, is_signed) — for BufferInfo compat."""
        for member in cls:
            if (member.bit_width == bit_width
                    and member.is_signed == is_signed
                    and not member.is_float):
                return member
        raise ValueError(f"No non-float DType for width={bit_width} signed={is_signed}")

    def const_to_bits(self, value) -> int:
        """Convert a Python numeric constant to an integer bit pattern.

        For float dtypes, produces the IEEE 754 bit pattern.
        For integer dtypes, masks to the appropriate width.
        """
        if self.is_float:
            if self.bit_width == 32:
                return struct.unpack(">I", struct.pack(">f", float(value)))[0]
            elif self.bit_width == 16:
                # fp16 — best effort via numpy if available, else truncate
                try:
                    import numpy as np
                    return int(np.float16(value).view(np.uint16))
                except ImportError:
                    return int(value) & 0xFFFF
        mask = (1 << self.bit_width) - 1
        return int(value) & mask


# ---------------------------------------------------------------------------
# IR value nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IRConst:
    """A compile-time constant scalar."""
    value: object   # int or float (Python native)
    dtype: DType


@dataclass(frozen=True)
class IRCounter:
    """A loop induction variable (the RANGE UOp)."""
    bound: int    # exclusive upper bound
    depth: int    # nesting depth, 0 = outermost loop


@dataclass(frozen=True)
class IRBufLoad:
    """A value loaded from a global buffer at a given address."""
    buf_idx: int
    addr: object    # IRValue (forward reference — Python allows this at runtime)


@dataclass(frozen=True)
class IRRegLoad:
    """A value read from the accumulator register."""
    dtype: DType


@dataclass(frozen=True)
class IROp:
    """An arithmetic or logical operation on one or more source IRValues."""
    op: str       # "add", "mul", "cmplt", "where", "max", "cast"
    dtype: DType  # result dtype
    srcs: tuple   # tuple of IRValue


# Type alias — the union of all value-producing IR nodes.
IRValue = Union[IRConst, IRCounter, IRBufLoad, IRRegLoad, IROp]


# ---------------------------------------------------------------------------
# IR store nodes (side-effecting)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IRBufStore:
    """Write a value to a global buffer at a given address."""
    buf_idx: int
    addr: object    # IRValue
    value: object   # IRValue
    dtype: DType


@dataclass(frozen=True)
class IRRegStore:
    """Write a value into the accumulator register."""
    value: object   # IRValue


# ---------------------------------------------------------------------------
# LoopIR — one level of a loop nest
# ---------------------------------------------------------------------------

@dataclass
class LoopIR:
    """One level of the loop tree.

    The root has axis_type=None, bound=0, depth=-1; its prologue/epilogue hold
    scalar-kernel stores.  Each nested level represents one RANGE/END pair.
    """
    axis_type: object            # tinygrad AxisType or None for root
    bound: int                   # iteration count (0 for root)
    depth: int                   # nesting depth (-1 for root, 0=outermost loop)
    prologue: list = field(default_factory=list)   # list[IRRegStore | IRBufStore]
    body: "LoopIR | None" = None
    epilogue: list = field(default_factory=list)   # list[IRRegStore | IRBufStore]


# ---------------------------------------------------------------------------
# BufferMeta — global buffer descriptor
# ---------------------------------------------------------------------------

@dataclass
class BufferMeta:
    """Typed descriptor for a DEFINE_GLOBAL buffer."""
    idx: int         # buffer index (arg of DEFINE_GLOBAL)
    depth: int       # number of elements
    dtype: DType     # element dtype
    is_output: bool  # True when idx == 0


# ---------------------------------------------------------------------------
# KernelIR — top-level typed kernel representation
# ---------------------------------------------------------------------------

@dataclass
class KernelIR:
    """Typed intermediate representation for one compiled kernel."""
    buffers: list               # list[BufferMeta]
    acc_dtype: "DType | None"   # None for kernels with no accumulator
    loop_tree: LoopIR           # root LoopIR (axis_type=None)
    scalar_stores: list = field(default_factory=list)  # list[IRBufStore] for no-loop kernels
    def format(self, kir: KernelIR, *, show_buffers: bool = True) -> str:
        """
        Tinygrad-ish dump of KernelIR.

        Produces something visually close to tinygrad's UOp table:
          - numbered statements
          - %value ids for loads/ops/counters
          - explicit RANGE/END
          - STORE lines in the loop prologue/epilogue order
        """
        # ---- value numbering (%0, %1, ...) ----
        val_ids: dict[int, int] = {}
        next_val = 0

        def vid(v: Any) -> str:
            nonlocal next_val
            if v is None:
                return "None"
            key = id(v)
            if key not in val_ids:
                val_ids[key] = next_val
                next_val += 1
            return f"%{val_ids[key]}"

        def dtype_str(dt: Any) -> str:
            # DType enum -> "int32"/"fp32"/...
            return getattr(dt, "name", str(dt)).lower()

        def buf_str(idx: int) -> str:
            return f"buf{idx}"

        def axis_str(ax: Any) -> str:
            return str(ax).replace("AxisType.", "")

        def val_expr(v: Any) -> str:
            # Prefer showing as "%id" with a short suffix so it reads like IR.
            if isinstance(v, IRConst):
                return f"{vid(v)}=CONST({v.value})<{dtype_str(v.dtype)}>"
            if isinstance(v, IRCounter):
                return f"{vid(v)}=COUNTER(d={v.depth}, bound={v.bound})<i32>"
            if isinstance(v, IRBufLoad):
                return f"{vid(v)}=LOAD({buf_str(v.buf_idx)}, {val_ref(v.addr)})"
            if isinstance(v, IRRegLoad):
                return f"{vid(v)}=RLOAD<{dtype_str(v.dtype)}>"
            if isinstance(v, IROp):
                srcs = ", ".join(val_ref(s) for s in v.srcs)
                return f"{vid(v)}={v.op.upper()}({srcs})<{dtype_str(v.dtype)}>"
            # fallback
            return f"{vid(v)}={type(v).__name__}"

        def val_ref(v: Any) -> str:
            # Reference to an already-emitted value. If not emitted yet, we still print %id
            # (like tinygrad does when a value appears before its defining row sometimes).
            if v is None:
                return "None"
            return vid(v)

        # ---- statement numbering (#0, #1, ...) ----
        stmts: list[str] = []
        stmt_no = 0

        def emit(indent: int, text: str) -> None:
            nonlocal stmt_no
            pad = "  " * indent
            stmts.append(f"{stmt_no:4d}  {pad}{text}")
            stmt_no += 1

        # ---- optional buffer header ----
        if show_buffers:
            stmts.append("BUFFERS:")
            for bm in kir.buffers:
                role = "out" if bm.is_output else "in"
                stmts.append(f"  - {buf_str(bm.idx)}: depth={bm.depth} dtype={dtype_str(bm.dtype)} role={role}")
            stmts.append("")

        # ---- ensure value definitions are emitted before use ----
        def ensure(v: Any, indent: int) -> None:
            if v is None:
                return
            if id(v) in val_ids:
                return
            if isinstance(v, (IRConst, IRCounter)):
                emit(indent, val_expr(v))
                return
            if isinstance(v, IRBufLoad):
                ensure(v.addr, indent)
                emit(indent, val_expr(v))
            elif isinstance(v, IROp):
                for s2 in v.srcs:
                    ensure(s2, indent)
                emit(indent, val_expr(v))
            elif isinstance(v, IRRegLoad):
                emit(indent, val_expr(v))
            else:
                emit(indent, val_expr(v))

        def ensure_store(s, indent: int) -> None:
            if isinstance(s, IRBufStore):
                ensure(s.addr, indent)
                ensure(s.value, indent)
            elif isinstance(s, IRRegStore):
                ensure(s.value, indent)

        def emit_store(s, indent: int) -> None:
            if isinstance(s, IRBufStore):
                emit(
                    indent,
                    f"STORE({buf_str(s.buf_idx)}, {val_ref(s.addr)}, {val_ref(s.value)})<{dtype_str(s.dtype)}>",
                )
            elif isinstance(s, IRRegStore):
                emit(indent, f"RSTORE({val_ref(s.value)})")
            else:
                emit(indent, f"{type(s).__name__}")

        # ---- loop tree walk ----
        def walk_loop(node: LoopIR, indent: int) -> None:
            # prologue
            for s in node.prologue:
                ensure_store(s, indent)
                emit_store(s, indent)

            # body loop
            if node.body is not None:
                lp = node.body
                emit(indent, f"RANGE {axis_str(lp.axis_type)} bound={lp.bound}")

                walk_loop(lp, indent + 1)

                emit(indent, "END")

            # epilogue
            for s in node.epilogue:
                ensure_store(s, indent)
                emit_store(s, indent)

        # Root-level (no explicit RANGE for root)
        stmts.append("KERNEL:")
        emit(0, "KERNEL_BEGIN")
        walk_loop(kir.loop_tree, 0)
        emit(0, "KERNEL_END")

        return "\n".join(stmts)

    def pretty(self) -> str:
        return self.format(self)

    def __str__(self):
        return self.pretty()

    def __repr__(self):
        return f"KernelIR(buffers={len(self.buffers)}, acc_dtype={self.acc_dtype})"
