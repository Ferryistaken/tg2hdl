# System Architecture

## Compiler pipeline

tinygrad's `schedule()` splits a model into compute kernels. Each kernel is a `list[UOp]` — tinygrad's linearized IR. `compile_kernel()` (`backend.py`) takes that list and returns a `CompiledKernel`: an Amaranth `Elaboratable` built in four passes.

```
tinygrad UOps
    │
    ├─ Pass 0: _create_memories()      — one Memory per DEFINE_GLOBAL buffer
    ├─ Pass 1: uop_to_ir()             — UOps → typed KernelIR
    ├─ Pass 2: ArithmeticLowering      — KernelIR → Amaranth combinational signals
    └─ Pass 3: build_control()         — KernelIR + signals → FSM
```

## Pass 0 — Memories (`hdl_module.py`)

Every `DEFINE_GLOBAL` buffer in the UOps becomes an Amaranth `Memory`:

- **Depth**: element count from `dtype.size`
- **Width**: determined by `DType.bit_width` (8 for int8, 16 for float16, 32 for int32/float32)
- **Ports**: two combinational read ports (one internal for the datapath, one external for inspection via `buf_read_ports`), one synchronous write port shared between external loading and FSM writes

Default write port wiring routes `buf{N}_wen/waddr/wdata` inputs to the internal write port. FSM states override this with `m.d.comb` when they need to write output.

## Pass 1 — Typed IR (`compiler/ir.py`, `compiler/uop_to_ir.py`)

`uop_to_ir()` walks the UOp list in a single pass and produces a `KernelIR` — a typed intermediate representation that makes all dtype and loop-structure information explicit.

### IR value nodes

```python
IRConst(value, dtype)              # compile-time constant
IRCounter(bound, depth)            # loop induction variable
IRBufLoad(buf_idx, addr: IRValue)  # load from a global buffer
IRRegLoad(dtype)                   # load from accumulator register
IROp(op, dtype, srcs: tuple)       # arithmetic/logical result
```

`IRValue` is the union of the above.  All nodes are frozen dataclasses — they are structural, immutable, and hashable.

### IR store nodes (side-effecting)

```python
IRRegStore(value: IRValue)                          # write into accumulator
IRBufStore(buf_idx, addr, value: IRValue, dtype)    # write into buffer
```

### Loop tree

```python
@dataclass
class LoopIR:
    axis_type: AxisType | None  # LOOP, REDUCE, or None (root)
    bound: int                  # iteration count (0 for root)
    depth: int                  # nesting depth (-1 for root, 0=outermost loop)
    prologue: list[IRRegStore | IRBufStore]
    body: LoopIR | None
    epilogue: list[IRRegStore | IRBufStore]
```

The root is always returned (axis_type=None, depth=-1). Its body is the outermost RANGE, or None for scalar kernels. Prologue stores (e.g. `acc = 0`) belong to the outer level; body stores (MAC accumulation) belong to the inner level; epilogue stores (output write) belong back to the outer level.

**LOOP vs REDUCE:** tinygrad tags each RANGE with an `AxisType`. `LOOP` axes are independent iterations (one output element per iteration — parallelizable in principle). `REDUCE` axes are dependent — each iteration reads and updates the accumulator. Both execute sequentially today; the distinction is preserved for future parallelism.

### DType

```python
class DType(enum.Enum):
    INT8, INT16, INT32, UINT8, UINT16, UINT32  # integer types
    FP16, BF16, FP32                            # float types (stored as IEEE bit patterns)

    def amaranth_shape(self) -> Shape:  ...     # signed/unsigned Amaranth shape
    def const_to_bits(self, value) -> int: ...  # float → IEEE 754 bit pattern
    def from_tinygrad(cls, tg_dtype) -> DType:  # raises ValueError on unsupported dtype
```

## Pass 2 — Arithmetic lowering (`compiler/lowering/arithmetic.py`)

`ArithmeticLowering.run()` traverses all `IRValue` nodes reachable from stores in the loop tree (in topological order, leaves first) and emits combinational Amaranth logic for each.

| IRValue type | Amaranth emission |
|---|---|
| `IRConst` | `Const(value)` — float constants converted to IEEE 754 bit patterns |
| `IRCounter` | counter `Signal` from `create_counters()` |
| `IRRegLoad` | the `acc` Signal |
| `IRBufLoad(buf_idx, addr)` | `m.d.comb += rp.addr.eq(addr_sig)`; new Signal wired to `rp.data` |
| `IROp("add"/"mul", FP32)` | `FP32Add`/`FP32Mul` submodule instantiated |
| `IROp("cmplt"/"max", FP32)` | `FP32Cmp` submodule instantiated |
| `IROp("add"/"mul", int)` | `Signal; m.d.comb += result.eq(a + b)` etc. |
| `IROp("cast")` | `Signal(dst_shape); m.d.comb += result.eq(src)` |
| `IROp("where")` | `Signal; m.d.comb += result.eq(Mux(cond, t, f))` |

The datapath is always live. Signals for loop addresses always reflect the current counter values, so when the FSM advances a counter, addresses and loaded data update automatically the next cycle.

### Float32 hardware modules (`compiler/fp32.py`)

Float32 operations use dedicated single-cycle combinational Amaranth modules that produce IEEE 754-accurate results in both simulation and synthesis:

- **`FP32Add`** — swap-by-magnitude → barrel-shift alignment → add/subtract → priority-encoder leading-one detection → left-shift normalization
- **`FP32Mul`** — XOR signs → 48-bit mantissa product (24×24) → normalize on `prod[47]` → 9-bit biased exponent arithmetic
- **`FP32Cmp`** — sign-aware magnitude comparison; handles ±0 (both-zero check), infinities, same/different sign cases

Known limitations: subnormal numbers flush to zero; rounding is truncation (round-toward-zero) rather than IEEE default round-to-nearest-even; `0 × ∞ = 0` rather than NaN.

## Pass 3 — FSM (`compiler/lowering/control.py`)

`build_control()` flattens the `LoopIR` tree to a list of `(level, depth)` pairs and creates Amaranth FSM states:

```
IDLE
  └─ on start: reset outermost counter → first state

L{d}_PRO  (non-innermost levels)
  └─ emit prologue stores (e.g. acc = 0)
     reset child counter → child's first state

L{d}_BODY  (innermost level)
  └─ emit body stores (e.g. acc += MAC result)
     if counter == bound-1: → parent EPI (or set done, → IDLE)
     else: counter++, → self

L{d}_EPI  (non-innermost levels)
  └─ emit epilogue stores (e.g. output[i] = cast(acc))
     if counter == bound-1: → parent EPI (or set done, → IDLE)
     else: counter++, → L{d}_PRO
```

**Store emission** (`_emit_stores`): each typed store node is resolved via `ArithResult.signals`:
- `IRRegStore(value)` → `m.d.sync += acc.eq(result.signals[id(value)])`
- `IRBufStore(buf_idx, addr, value)` → `m.d.comb += [wp.addr.eq(...), wp.data.eq(...), wp.en.eq(1)]` (overrides default external wiring)

**Done signal:** set synchronously (`m.d.sync += done.eq(1)`) in the last compute state when the outermost counter reaches its bound, then cleared in IDLE. Done is valid for exactly one cycle after the final store.

## Timing model

| Pattern | States | Cycle count |
|---------|--------|-------------|
| Scalar (no RANGE) | IDLE → SCALAR → IDLE | 1 |
| Elementwise (single LOOP, bound N) | IDLE → L0_BODY × N → IDLE | N |
| GEMV (LOOP M, REDUCE K) | IDLE → (L0_PRO + L1_BODY×K + L0_EPI) × M → IDLE | M×(K+2) |

## Overflow analysis

$$\max(\text{acc}) = K \times 127^2 = K \times 16{,}129$$

For K=784: 12,645,136 < 2³¹−1 ✓

## Multi-kernel hardware (`compiler/top_module.py`)

`TopModule` sequences N `CompiledKernel` instances with a hardware copy FSM:

```
IDLE → K0_RUN → K0_WAIT → COPY_0_1 → K1_RUN → K1_WAIT → ... → DONE
```

The copy FSM reads from the source kernel's output buffer and writes to the destination kernel's input buffer, one element per cycle. `compile_top_module(schedule)` auto-detects connections by checking tinygrad Buffer object identity across schedule items.
