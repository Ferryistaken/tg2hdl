"""UOp → KernelIR conversion.

Converts a linearized tinygrad UOp list into a typed KernelIR, merging
the roles of the old _parse_loop_structure and value-analysis portion of
_build_datapath into one sequential pass.

No Amaranth imports — this module is purely structural/analytical.
"""

from tinygrad.uop.ops import Ops
from tinygrad.dtype import AddrSpace

from .ir import (
    DType,
    IRConst, IRCounter, IRBufLoad, IRRegLoad, IROp,
    IRBufStore, IRRegStore,
    LoopIR, BufferMeta, KernelIR,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_ptr(ptr_uop) -> int:
    """Walk AFTER chains to find the DEFINE_GLOBAL buf index.

    Returns -1 for DEFINE_REG (register access).
    Returns -2 if resolution fails.
    """
    current = ptr_uop
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if current.op == Ops.DEFINE_GLOBAL:
            return current.arg
        if current.op == Ops.DEFINE_REG:
            return -1
        if current.op == Ops.AFTER and current.src:
            current = current.src[0]
            continue
        break
    return -2


def _try_dtype(tg_dtype, fallback: DType) -> DType:
    """Try to convert a tinygrad dtype, returning fallback on failure."""
    try:
        return DType.from_tinygrad(tg_dtype)
    except (ValueError, AttributeError, TypeError):
        return fallback


def _irvalue_dtype(val, buf_meta_map: dict, fallback: DType) -> DType:
    """Extract the DType from any IRValue."""
    if isinstance(val, IRConst):
        return val.dtype
    if isinstance(val, IRCounter):
        return DType.INT32  # counters are always integer indices
    if isinstance(val, IRBufLoad):
        meta = buf_meta_map.get(val.buf_idx)
        return meta.dtype if meta else fallback
    if isinstance(val, IRRegLoad):
        return val.dtype
    if isinstance(val, IROp):
        return val.dtype
    return fallback


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def uop_to_ir(uops, buf_metas: list) -> KernelIR:
    """Convert a linearized UOp list to a typed KernelIR.

    Parameters
    ----------
    uops : list[UOp]
        Linearized UOps from tinygrad (via linearize()).
    buf_metas : list[BufferMeta]
        Buffer descriptors pre-built from DEFINE_GLOBAL analysis.

    Returns
    -------
    KernelIR
    """
    buf_meta_map = {bm.idx: bm for bm in buf_metas}

    # val_map: id(uop) → IRValue (or None for non-value UOps)
    val_map: dict = {}
    acc_dtype: DType | None = None

    # Loop tree construction
    root = LoopIR(axis_type=None, bound=0, depth=-1)
    stack = [root]
    mode_stack = ["before"]

    for u in uops:

        # ----------------------------------------------------------------
        # Buffer / register declarations
        # ----------------------------------------------------------------
        if u.op == Ops.DEFINE_GLOBAL:
            # Buffer pointer — not a value, tracked separately in buf_metas
            val_map[id(u)] = None
            continue

        elif u.op == Ops.DEFINE_REG:
            # Accumulator register declaration
            acc_dtype = _try_dtype(u.dtype, DType.INT32)
            val_map[id(u)] = IRRegLoad(acc_dtype)
            continue

        # ----------------------------------------------------------------
        # Constants
        # ----------------------------------------------------------------
        elif u.op == Ops.CONST:
            dtype = _try_dtype(u.dtype, DType.INT32)
            val_map[id(u)] = IRConst(u.arg, dtype)
            continue

        # ----------------------------------------------------------------
        # Loop control
        # ----------------------------------------------------------------
        elif u.op == Ops.RANGE:
            # depth = number of loops already on stack (root is depth -1,
            # first real loop is depth 0)
            cur_depth = len(stack) - 1  # 0 for outermost loop
            bound = u.src[0].arg
            parent = stack[-1]
            child = LoopIR(
                axis_type=u.arg[1],
                bound=bound,
                depth=cur_depth,
            )
            parent.body = child
            stack.append(child)
            mode_stack.append("before")

            val_map[id(u)] = IRCounter(bound=bound, depth=cur_depth)
            continue

        elif u.op == Ops.END:
            stack.pop()
            mode_stack.pop()
            if mode_stack:
                mode_stack[-1] = "after"
            continue

        elif u.op == Ops.AFTER:
            # AFTER is an ordering barrier; propagate source value
            val_map[id(u)] = val_map.get(id(u.src[0])) if u.src else None
            continue

        elif u.op in (Ops.SINK,):
            continue

        # ----------------------------------------------------------------
        # Memory access
        # ----------------------------------------------------------------
        elif u.op == Ops.INDEX:
            ptr_uop = u.src[0]
            offset_uop = u.src[1]

            # Determine register vs buffer access
            if hasattr(u.dtype, "addrspace") and u.dtype.addrspace == AddrSpace.REG:
                # Register index (accumulator)
                dtype = acc_dtype if acc_dtype is not None else DType.INT32
                val_map[id(u)] = IRRegLoad(dtype)
            else:
                buf_idx = _resolve_ptr(ptr_uop)
                addr_expr = val_map.get(id(offset_uop), IRConst(0, DType.INT32))
                val_map[id(u)] = IRBufLoad(buf_idx=buf_idx, addr=addr_expr)
            continue

        elif u.op == Ops.LOAD:
            # LOAD propagates its source INDEX's IRValue
            src_val = val_map.get(id(u.src[0]))
            if isinstance(src_val, IRRegLoad):
                # Register load — return same IRRegLoad
                val_map[id(u)] = src_val
            elif isinstance(src_val, IRBufLoad):
                # Buffer load — return same IRBufLoad (lowering will create signal)
                val_map[id(u)] = src_val
            else:
                raise ValueError(
                    f"uop_to_ir: LOAD at UOp {u!r} has an unresolvable source "
                    f"(resolved to {src_val!r}). Expected IRRegLoad or IRBufLoad. "
                    "This indicates the preceding INDEX UOp was not correctly handled."
                )
            continue

        elif u.op == Ops.STORE:
            # Resolve target (INDEX) and value (src[1])
            index_val = val_map.get(id(u.src[0]))
            value_val = val_map.get(id(u.src[1]))

            if value_val is None:
                raise ValueError(
                    f"uop_to_ir: STORE at UOp {u!r} has an unresolvable value operand "
                    f"(src[1] resolved to None in val_map). "
                    "The value UOp must produce a valid IRValue."
                )

            # Build typed store node
            store_node: IRBufStore | IRRegStore
            if isinstance(index_val, IRRegLoad):
                store_node = IRRegStore(value=value_val)
            elif isinstance(index_val, IRBufLoad):
                # Use the buffer's meta dtype for the store
                meta = buf_meta_map.get(index_val.buf_idx)
                dtype = meta.dtype if meta else _irvalue_dtype(value_val, buf_meta_map, DType.INT32)
                store_node = IRBufStore(
                    buf_idx=index_val.buf_idx,
                    addr=index_val.addr,
                    value=value_val,
                    dtype=dtype,
                )
            else:
                raise ValueError(
                    f"uop_to_ir: STORE at UOp {u!r} has an unresolvable INDEX target "
                    f"(resolved to {index_val!r}). Expected IRRegLoad or IRBufLoad. "
                    "This indicates the preceding INDEX UOp was not correctly handled."
                )

            # Place store in current loop level's prologue or epilogue
            cur = stack[-1]
            mode = mode_stack[-1]
            if mode == "before":
                cur.prologue.append(store_node)
            else:
                cur.epilogue.append(store_node)

            val_map[id(u)] = None
            continue

        # ----------------------------------------------------------------
        # Arithmetic / logical ops
        # ----------------------------------------------------------------
        elif u.op == Ops.MUL:
            dtype = _try_dtype(u.dtype, acc_dtype or DType.INT32)
            a = val_map.get(id(u.src[0]))
            b = val_map.get(id(u.src[1]))
            val_map[id(u)] = IROp("mul", dtype, (a, b))
            continue

        elif u.op == Ops.ADD:
            dtype = _try_dtype(u.dtype, acc_dtype or DType.INT32)
            a = val_map.get(id(u.src[0]))
            b = val_map.get(id(u.src[1]))
            val_map[id(u)] = IROp("add", dtype, (a, b))
            continue

        elif u.op == Ops.CAST:
            dtype = _try_dtype(u.dtype, acc_dtype or DType.INT32)
            src = val_map.get(id(u.src[0]))
            val_map[id(u)] = IROp("cast", dtype, (src,))
            continue

        elif u.op == Ops.CMPLT:
            # Result is boolean (1-bit), stored as UINT8 for simplicity
            a = val_map.get(id(u.src[0]))
            b = val_map.get(id(u.src[1]))
            # Record operand dtype on the IROp so dispatch can check is_float
            op_dtype = _irvalue_dtype(a, buf_meta_map, DType.INT32) if a is not None else DType.INT32
            val_map[id(u)] = IROp("cmplt", op_dtype, (a, b))
            continue

        elif u.op == Ops.WHERE:
            dtype = _try_dtype(u.dtype, acc_dtype or DType.INT32)
            cond = val_map.get(id(u.src[0]))
            t_val = val_map.get(id(u.src[1]))
            f_val = val_map.get(id(u.src[2]))
            val_map[id(u)] = IROp("where", dtype, (cond, t_val, f_val))
            continue

        elif u.op == Ops.MAX:
            dtype = _try_dtype(u.dtype, acc_dtype or DType.INT32)
            a = val_map.get(id(u.src[0]))
            b = val_map.get(id(u.src[1]))
            val_map[id(u)] = IROp("max", dtype, (a, b))
            continue

        else:
            raise NotImplementedError(
                f"uop_to_ir: unsupported UOp {u.op!r} (dtype={getattr(u, 'dtype', '?')}). "
                "Add a handler in uop_to_ir() before use."
            )

    # Scalar stores live in root.prologue / root.epilogue (no-loop kernels)
    scalar_stores = [
        s for s in (root.prologue + root.epilogue)
        if isinstance(s, IRBufStore)
    ]

    return KernelIR(
        buffers=list(buf_metas),
        acc_dtype=acc_dtype,
        loop_tree=root,
        scalar_stores=scalar_stores,
    )
