"""Arithmetic lowering: KernelIR → Amaranth Signals (combinational datapath).

Converts typed IRValue nodes to Amaranth Signal objects, emitting
combinational assignments into the provided Module.  Float operations
dispatch to the IEEE 754 FP32 hardware modules in compiler/fp32.py.

No FSM logic — that is the responsibility of control.py.
"""

import struct
from dataclasses import dataclass, field
from typing import Any

from amaranth.hdl import Module, Signal, signed, unsigned, Const, Mux, Cat

from ..ir import (
    DType,
    IRConst, IRCounter, IRBufLoad, IRRegLoad, IROp,
    IRBufStore, IRRegStore,
    LoopIR, BufferMeta, KernelIR,
)
from ..fp32 import FP32Add, FP32Mul, FP32Cmp, FP32Reciprocal


# ---------------------------------------------------------------------------
# ArithResult
# ---------------------------------------------------------------------------

@dataclass
class ArithResult:
    """Holds the Amaranth Signal (or Const) for each IRValue in the kernel.

    Keyed by id(IRValue).  A few special entries:
      - result.acc : the accumulator Signal (or None)
      - result.counter_sigs[depth] : Signal for loop counter at depth d
    """
    signals: dict = field(default_factory=dict)   # id(IRValue) → Amaranth Signal/Const
    acc: Any = None                                # accumulator Signal
    counter_sigs: dict = field(default_factory=dict)  # depth → Signal


# ---------------------------------------------------------------------------
# create_counters — extracted helper used before ArithmeticLowering
# ---------------------------------------------------------------------------

def create_counters(kernel: KernelIR, m: Module) -> dict:
    """Walk the LoopIR tree and create an Amaranth Signal for each loop level.

    Returns a dict mapping loop depth (int) → Signal.
    """
    counter_sigs = {}
    level = kernel.loop_tree.body
    while level is not None:
        d = level.depth
        ctr = Signal(range(max(level.bound, 1)), name=f"ctr_L{d}")
        counter_sigs[d] = ctr
        level = level.body
    return counter_sigs


# ---------------------------------------------------------------------------
# ArithmeticLowering
# ---------------------------------------------------------------------------

class ArithmeticLowering:
    """Convert typed IRValues to Amaranth combinational logic.

    Usage
    -----
    counter_sigs = create_counters(kernel_ir, m)
    acc = Signal(kernel_ir.acc_dtype.amaranth_shape(), name="acc") if kernel_ir.acc_dtype else None
    lowering = ArithmeticLowering(kernel_ir, m, int_rports, counter_sigs, acc)
    result = lowering.run()
    """

    def __init__(
        self,
        kernel: KernelIR,
        m: Module,
        int_rports: dict,       # buf_idx → Amaranth read port
        counter_sigs: dict,     # depth → Signal (from create_counters)
        acc,                    # Amaranth Signal or None
        unroll_factor: int = 1,
    ):
        self.kernel = kernel
        self.m = m
        self.int_rports = int_rports
        self.counter_sigs = counter_sigs
        self.acc = acc
        self.unroll_factor = max(1, int(unroll_factor))
        self._buf_read_use = {}

        # Build buf_meta_map for quick lookup
        self.buf_meta_map = {bm.idx: bm for bm in kernel.buffers}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> ArithResult:
        """Walk all IRValues referenced in the kernel and emit combinational logic.

        Traversal order: collect all IRValues reachable from stores in the
        loop tree, then emit in topological order (leaves first).
        """
        result = ArithResult(acc=self.acc, counter_sigs=self.counter_sigs)

        # Collect all unique IRValues in the kernel (via store chains)
        all_vals = []
        visited_ids = set()

        def collect(val):
            if val is None or id(val) in visited_ids:
                return
            visited_ids.add(id(val))
            # Recurse into sources first (topological order: leaves → roots)
            if isinstance(val, IROp):
                for src in val.srcs:
                    collect(src)
            elif isinstance(val, IRBufLoad):
                collect(val.addr)
            all_vals.append(val)

        # Collect from all stores in the loop tree
        self._collect_stores(self.kernel.loop_tree, collect)
        for s in self.kernel.scalar_stores:
            collect(s.value)
            collect(s.addr)

        # Emit combinational logic for each IRValue
        uid_counter = [0]
        for val in all_vals:
            self._emit(val, result, uid_counter)

        return result

    def _collect_stores(self, level: LoopIR, collect_fn):
        """Recursively collect IRValues from all stores in the loop tree."""
        for store in level.prologue + level.epilogue:
            if isinstance(store, (IRBufStore, IRRegStore)):
                collect_fn(store.value)
            if isinstance(store, IRBufStore):
                collect_fn(store.addr)
        if level.body is not None:
            self._collect_stores(level.body, collect_fn)

    # ------------------------------------------------------------------
    # Signal emission per IRValue type
    # ------------------------------------------------------------------

    def _emit(self, val, result: ArithResult, uid_ctr: list):
        """Emit combinational logic for one IRValue. Idempotent (no-op if done)."""
        if id(val) in result.signals:
            return  # already emitted

        uid = uid_ctr[0]
        uid_ctr[0] += 1
        m = self.m

        if isinstance(val, IRConst):
            if val.dtype.is_float:
                bits = val.dtype.const_to_bits(val.value)
                result.signals[id(val)] = Const(bits)
            else:
                result.signals[id(val)] = Const(int(val.value))

        elif isinstance(val, IRCounter):
            sig = self.counter_sigs.get(val.depth)
            if sig is None:
                raise RuntimeError(
                    f"ArithmeticLowering: no counter signal for loop depth {val.depth}. "
                    "Ensure create_counters() was called with the correct KernelIR before "
                    "constructing ArithmeticLowering."
                )
            result.signals[id(val)] = sig

        elif isinstance(val, IRRegLoad):
            if self.acc is None:
                raise RuntimeError(
                    "ArithmeticLowering: IRRegLoad encountered but the accumulator signal "
                    "is None. KernelIR has acc_dtype=None but the IR contains a register "
                    "load — the UOp graph references a register that was never declared."
                )
            result.signals[id(val)] = self.acc

        elif isinstance(val, IRBufLoad):
            addr_sig = result.signals.get(id(val.addr))
            if addr_sig is None:
                raise RuntimeError(
                    f"ArithmeticLowering: IRBufLoad(buf_idx={val.buf_idx}) has no emitted "
                    "signal for its address expression. This indicates a topological "
                    "ordering failure in run() — the address IRValue was not collected "
                    "before the IRBufLoad that references it."
                )
            rps = self.int_rports.get(val.buf_idx)
            if not rps:
                raise RuntimeError(
                    f"ArithmeticLowering: IRBufLoad references buf_idx={val.buf_idx} "
                    "but no read port exists for that buffer index. "
                    "Check that _create_memories() created a port for every buffer."
                )
            use_idx = self._buf_read_use.get(val.buf_idx, 0)
            if use_idx >= len(rps):
                raise RuntimeError(
                    f"ArithmeticLowering: buf_idx={val.buf_idx} needs {use_idx + 1} read ports "
                    f"but only {len(rps)} were created."
                )
            rp = rps[use_idx]
            self._buf_read_use[val.buf_idx] = use_idx + 1
            # Wire the read port address
            m.d.comb += rp.addr.eq(addr_sig)
            # Create a signal to hold the read data
            meta = self.buf_meta_map.get(val.buf_idx)
            shape = meta.dtype.amaranth_shape() if meta else unsigned(32)
            load_sig = Signal(shape, name=f"load_b{val.buf_idx}_{uid}")
            m.d.comb += load_sig.eq(rp.data)
            result.signals[id(val)] = load_sig

        elif isinstance(val, IROp):
            self._emit_op(val, result, uid, uid_ctr)

    def _emit_op(self, val: IROp, result: ArithResult, uid: int, uid_ctr: list):
        """Emit combinational logic for an IROp node."""
        m = self.m

        def get_sig(src):
            if src is None:
                raise RuntimeError(
                    f"ArithmeticLowering: IROp {val!r} has None as a source — "
                    "a UOp dependency was not resolved in uop_to_ir(). "
                    "Check that all UOp sources are handled before their consumers."
                )
            sig = result.signals.get(id(src))
            if sig is None:
                raise RuntimeError(
                    f"ArithmeticLowering: source signal for {src!r} (used by IROp "
                    f"{val.op!r}) was not emitted before its consumer. "
                    "This indicates a topological ordering failure in run()."
                )
            return sig

        op = val.op
        dtype = val.dtype

        if op == "mul":
            a, b = val.srcs[0], val.srcs[1]
            a_sig = get_sig(a)
            b_sig = get_sig(b)
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"mul_{uid}")
            if dtype.is_float and dtype.bit_width == 32:
                fp = FP32Mul(uid=uid)
                m.submodules[f"fp_mul_{uid}"] = fp
                m.d.comb += [fp.a.eq(a_sig), fp.b.eq(b_sig), result_sig.eq(fp.result)]
            else:
                m.d.comb += result_sig.eq(a_sig * b_sig)
            result.signals[id(val)] = result_sig

        elif op == "add":
            a, b = val.srcs[0], val.srcs[1]
            a_sig = get_sig(a)
            b_sig = get_sig(b)
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"add_{uid}")
            if dtype.is_float and dtype.bit_width == 32:
                fp = FP32Add(uid=uid)
                m.submodules[f"fp_add_{uid}"] = fp
                m.d.comb += [fp.a.eq(a_sig), fp.b.eq(b_sig), result_sig.eq(fp.result)]
            else:
                m.d.comb += result_sig.eq(a_sig + b_sig)
            result.signals[id(val)] = result_sig

        elif op == "cmplt":
            a, b = val.srcs[0], val.srcs[1]
            a_sig = get_sig(a)
            b_sig = get_sig(b)
            result_sig = Signal(name=f"cmplt_{uid}")
            if dtype.is_float and dtype.bit_width == 32:
                fp = FP32Cmp(uid=uid)
                m.submodules[f"fp_cmp_{uid}"] = fp
                m.d.comb += [fp.a.eq(a_sig), fp.b.eq(b_sig), result_sig.eq(fp.result)]
            else:
                m.d.comb += result_sig.eq(a_sig < b_sig)
            result.signals[id(val)] = result_sig

        elif op == "max":
            a, b = val.srcs[0], val.srcs[1]
            a_sig = get_sig(a)
            b_sig = get_sig(b)
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"max_{uid}")
            if dtype.is_float and dtype.bit_width == 32:
                # max(a, b) = if a < b then b else a
                fp_cmp = FP32Cmp(uid=uid)
                m.submodules[f"fp_max_cmp_{uid}"] = fp_cmp
                m.d.comb += [fp_cmp.a.eq(a_sig), fp_cmp.b.eq(b_sig)]
                m.d.comb += result_sig.eq(Mux(fp_cmp.result, b_sig, a_sig))
            else:
                m.d.comb += result_sig.eq(Mux(a_sig > b_sig, a_sig, b_sig))
            result.signals[id(val)] = result_sig

        elif op == "where":
            cond, t_val, f_val = val.srcs[0], val.srcs[1], val.srcs[2]
            cond_sig = get_sig(cond)
            t_sig = get_sig(t_val)
            f_sig = get_sig(f_val)
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"where_{uid}")
            m.d.comb += result_sig.eq(Mux(cond_sig, t_sig, f_sig))
            result.signals[id(val)] = result_sig

        elif op == "cast":
            src = val.srcs[0]
            src_sig = get_sig(src)
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"cast_{uid}")

            src_dtype = src.dtype if hasattr(src, "dtype") else DType.INT32

            if src_dtype.is_float and not dtype.is_float:
                # float32 → int: truncate toward zero
                f_s   = src_sig[31]
                f_e   = src_sig[23:31]
                f_mnt = src_sig[0:23]
                full_mant_c = Signal(unsigned(24), name=f"cast_fmnt_{uid}")
                abs_int_c   = Signal(unsigned(32), name=f"cast_abs_{uid}")
                m.d.comb += full_mant_c.eq((1 << 23) | f_mnt)
                with m.Switch(f_e):
                    for ea_val in range(256):
                        with m.Case(ea_val):
                            ue = ea_val - 127
                            if ea_val == 0 or ue < 0:
                                m.d.comb += abs_int_c.eq(0)
                            elif ue > 30:
                                m.d.comb += abs_int_c.eq(0x7FFFFFFF)
                            elif ue >= 23:
                                m.d.comb += abs_int_c.eq(full_mant_c << (ue - 23))
                            else:
                                m.d.comb += abs_int_c.eq(full_mant_c >> (23 - ue))
                with m.If(f_s):
                    m.d.comb += result_sig.eq(-abs_int_c)
                with m.Else():
                    m.d.comb += result_sig.eq(abs_int_c)

            elif not src_dtype.is_float and dtype.is_float:
                # int → float32: convert integer value to IEEE 754
                int_s   = src_sig[31]
                abs_val = Signal(unsigned(32), name=f"cast_av_{uid}")
                with m.If(int_s):
                    m.d.comb += abs_val.eq(-src_sig)
                with m.Else():
                    m.d.comb += abs_val.eq(src_sig)
                lo = Signal(range(32), name=f"cast_lo_{uid}")
                m.d.comb += lo.eq(0)
                for i in range(32):
                    with m.If(abs_val[i]):
                        m.d.comb += lo.eq(i)
                shift_r = Signal(range(32), name=f"cast_sr_{uid}")
                shift_l = Signal(range(32), name=f"cast_sl_{uid}")
                right_s = Signal(unsigned(32), name=f"cast_rs_{uid}")
                left_s  = Signal(unsigned(32), name=f"cast_ls_{uid}")
                res_e_c = Signal(unsigned(8),  name=f"cast_re_{uid}")
                res_m_c = Signal(unsigned(23), name=f"cast_rm_{uid}")
                m.d.comb += [
                    shift_r.eq(Mux(lo >= 23, lo - 23, 0)),
                    shift_l.eq(Mux(lo >= 23, 0, 23 - lo)),
                    right_s.eq(abs_val >> shift_r),
                    left_s.eq(abs_val << shift_l),
                ]
                with m.If(abs_val == 0):
                    m.d.comb += [res_e_c.eq(0), res_m_c.eq(0)]
                with m.Elif(lo >= 23):
                    m.d.comb += [res_e_c.eq(lo + 127), res_m_c.eq(right_s[0:23])]
                with m.Else():
                    m.d.comb += [res_e_c.eq(lo + 127), res_m_c.eq(left_s[0:23])]
                m.d.comb += result_sig.eq(Cat(res_m_c, res_e_c, int_s))

            else:
                # bool→int, int↔int, same-type: bit extend / truncate
                m.d.comb += result_sig.eq(src_sig)

            result.signals[id(val)] = result_sig

        # ----------------------------------------------------------------
        # Easy ops — combinational integer arithmetic / bitwise
        # ----------------------------------------------------------------

        elif op == "sub":
            a_sig, b_sig = get_sig(val.srcs[0]), get_sig(val.srcs[1])
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"sub_{uid}")
            if dtype.is_float:
                raise NotImplementedError(
                    "ArithmeticLowering: float SUB not yet implemented — "
                    "needs an FP32Sub module. Use ADD(a, NEG(b)) as a workaround."
                )
            m.d.comb += result_sig.eq(a_sig - b_sig)
            result.signals[id(val)] = result_sig

        elif op == "neg":
            a_sig = get_sig(val.srcs[0])
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"neg_{uid}")
            if dtype.is_float and dtype.bit_width == 32:
                # Flip IEEE 754 sign bit — no arithmetic needed
                m.d.comb += result_sig.eq(a_sig ^ 0x80000000)
            elif dtype.is_float:
                raise NotImplementedError(
                    f"ArithmeticLowering: NEG not implemented for {dtype}."
                )
            else:
                m.d.comb += result_sig.eq(-a_sig)
            result.signals[id(val)] = result_sig

        elif op == "reciprocal":
            a_sig = get_sig(val.srcs[0])
            result_sig = Signal(unsigned(32), name=f"rcp_{uid}")
            if dtype.is_float and dtype.bit_width == 32:
                fp = FP32Reciprocal(uid=uid)
                m.submodules[f"fp_rcp_{uid}"] = fp
                m.d.comb += [fp.a.eq(a_sig), result_sig.eq(fp.result)]
            else:
                raise NotImplementedError(
                    f"ArithmeticLowering: RECIPROCAL only supported for FP32, got {dtype}."
                )
            result.signals[id(val)] = result_sig

        elif op == "and":
            a_sig, b_sig = get_sig(val.srcs[0]), get_sig(val.srcs[1])
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"and_{uid}")
            m.d.comb += result_sig.eq(a_sig & b_sig)
            result.signals[id(val)] = result_sig

        elif op == "or":
            a_sig, b_sig = get_sig(val.srcs[0]), get_sig(val.srcs[1])
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"or_{uid}")
            m.d.comb += result_sig.eq(a_sig | b_sig)
            result.signals[id(val)] = result_sig

        elif op == "xor":
            a_sig, b_sig = get_sig(val.srcs[0]), get_sig(val.srcs[1])
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"xor_{uid}")
            m.d.comb += result_sig.eq(a_sig ^ b_sig)
            result.signals[id(val)] = result_sig

        elif op == "shl":
            a_sig, b_sig = get_sig(val.srcs[0]), get_sig(val.srcs[1])
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"shl_{uid}")
            m.d.comb += result_sig.eq(a_sig << b_sig)
            result.signals[id(val)] = result_sig

        elif op == "shr":
            a_sig, b_sig = get_sig(val.srcs[0]), get_sig(val.srcs[1])
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"shr_{uid}")
            # Amaranth's >> on signed signals is arithmetic (sign-extending)
            m.d.comb += result_sig.eq(a_sig >> b_sig)
            result.signals[id(val)] = result_sig

        elif op == "idiv":
            a_sig, b_sig = get_sig(val.srcs[0]), get_sig(val.srcs[1])
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"idiv_{uid}")
            m.d.comb += result_sig.eq(a_sig // b_sig)
            result.signals[id(val)] = result_sig

        elif op == "mod":
            a_sig, b_sig = get_sig(val.srcs[0]), get_sig(val.srcs[1])
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"mod_{uid}")
            m.d.comb += result_sig.eq(a_sig % b_sig)
            result.signals[id(val)] = result_sig

        elif op == "cmpeq":
            a_sig, b_sig = get_sig(val.srcs[0]), get_sig(val.srcs[1])
            result_sig = Signal(name=f"cmpeq_{uid}")
            # Bit-pattern equality works for both int and float (no NaN in our model)
            m.d.comb += result_sig.eq(a_sig == b_sig)
            result.signals[id(val)] = result_sig

        elif op == "cmpne":
            a_sig, b_sig = get_sig(val.srcs[0]), get_sig(val.srcs[1])
            result_sig = Signal(name=f"cmpne_{uid}")
            m.d.comb += result_sig.eq(a_sig != b_sig)
            result.signals[id(val)] = result_sig

        elif op == "trunc":
            src = val.srcs[0]
            src_sig = get_sig(src)
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"trunc_{uid}")
            if dtype.is_float:
                raise NotImplementedError(
                    "ArithmeticLowering: float TRUNC (round-toward-zero to int) "
                    "not yet implemented — needs a dedicated FP→int conversion unit."
                )
            # For integer→integer, TRUNC is equivalent to a cast (truncate high bits)
            m.d.comb += result_sig.eq(src_sig)
            result.signals[id(val)] = result_sig

        elif op == "bitcast":
            # Reinterpret bit pattern as a different type — identical to cast at RTL level
            src = val.srcs[0]
            src_sig = get_sig(src)
            shape = dtype.amaranth_shape()
            result_sig = Signal(shape, name=f"bitcast_{uid}")
            m.d.comb += result_sig.eq(src_sig)
            result.signals[id(val)] = result_sig

        else:
            raise NotImplementedError(
                f"ArithmeticLowering: unsupported op {op!r} in IROp. "
                "Add a handler in _emit_op()."
            )
