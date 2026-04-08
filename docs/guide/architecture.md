# System Architecture

## Compiler pipeline

tinygrad's `schedule()` splits a model into compute kernels. Each kernel is a `list[UOp]` — tinygrad's linearized IR. `compile_kernel()` (`backend.py`) first converts UOps to a typed `KernelIR` via `uop_to_ir()`, then passes it to `CompiledKernel` which lowers it to Amaranth hardware in three passes.

```
compile_kernel()  (backend.py)
    │
    ├─ uop_to_ir()                     — UOps → typed KernelIR
    │
    └─ CompiledKernel.elaborate()      — KernelIR → Amaranth Module
        ├─ Pass 0: _create_memories()      — one Memory per DEFINE_GLOBAL buffer
        ├─ Pass 1: ArithmeticLowering      — KernelIR → Amaranth combinational signals
        └─ Pass 2: build_control()         — KernelIR + signals → FSM
```

## Pass 0 — Memories (`hdl_module.py`)

Every `DEFINE_GLOBAL` buffer in the UOps becomes an Amaranth `Memory`:

- **Depth**: element count from `dtype.size`
- **Width**: determined by `DType.bit_width` (8 for int8, 16 for float16, 32 for int32/float32)
- **Ports**: two combinational read ports (one internal for the datapath, one external for inspection via `buf_read_ports`), one synchronous write port shared between external loading and FSM writes

Default write port wiring routes `buf{N}_wen/waddr/wdata` inputs to the internal write port. FSM states override this with `m.d.comb` when they need to write output.

## Typed IR (`compiler/ir.py`, `compiler/uop_to_ir.py`)

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

## Pass 1 — Arithmetic lowering (`compiler/lowering/arithmetic.py`)

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

## Pass 2 — FSM (`compiler/lowering/control.py`)

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

`TopModule` sequences N `CompiledKernel` instances with a dependency-driven copy FSM. It supports arbitrary kernel DAGs — linear chains, skip connections (non-adjacent producer→consumer), and fan-out (one producer feeding multiple consumers).

### Execution model

```
IDLE → K0_RUN → K0_WAIT → K0_COPY_0 → K1_RUN → K1_WAIT → K1_COPY_0 → K2_RUN → K2_WAIT → IDLE
```

After each kernel finishes, the FSM runs one copy state per *source buffer group* — connections sharing the same producer buffer are grouped and broadcast in a single pass (same read address, N simultaneous writes). A two-kernel chain has one copy state; a fan-out from K0 to both K1 and K2 also has one copy state, with K0's output buffer read once and written to both destinations in parallel.

### Connection detection and ordering

`compile_top_module(schedule)` detects connections by checking tinygrad Buffer object identity: if a buffer appears as the output of kernel A and as an input of kernel B, an edge `(A, 0, B, j)` is recorded. Any non-self edge is accepted — non-adjacent and skip connections are handled correctly.

Kernels are then reordered by `_toposort_kernels()` (Kahn's algorithm) so that every producer precedes its consumers, regardless of the order tinygrad's scheduler emitted them. A cycle in the dependency graph raises `ValueError` at compile time.

### Copy groups

`_build_copy_groups()` partitions all connections by `(src_kernel, src_buffer)`. Each group becomes one FSM state `K{i}_COPY_{gi}`. Within a state, `copy_ctr` drives `src_rp["raddr"]` and all destination write ports simultaneously:

```
K{i}_COPY_0:
    src_rp.raddr  ← copy_ctr
    dst1_wp.waddr ← copy_ctr,  dst1_wp.wdata ← src_rp.rdata,  dst1_wp.wen ← 1
    dst2_wp.waddr ← copy_ctr,  dst2_wp.wdata ← src_rp.rdata,  dst2_wp.wen ← 1
    ...
    if copy_ctr == depth-1: copy_ctr ← 0, next state
    else: copy_ctr++
```

Copy length is taken from `buf_depths[(src_k, src_buf)]` — the source buffer size — which is correct for all destinations since they all receive the same data.

### External write ports

All input buffers not driven by any copy FSM edge are exposed as `ext_write_ports[(k_idx, buf_idx)]` — the testbench or simulation harness writes weights and inputs through these before pulsing `start`.
