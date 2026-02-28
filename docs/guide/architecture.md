# System Architecture

## Compiler pipeline

tinygrad's `schedule()` splits a model into compute kernels. Each kernel is a `list[UOp]` — tinygrad's linearized IR. `compile_kernel()` (`backend.py:102`) takes that list and returns a `CompiledKernel`: an Amaranth `Elaboratable` built in three passes.

```
tinygrad UOps
    │
    ├─ Pass 0: _create_memories()   — one Memory per DEFINE_GLOBAL
    ├─ Pass 1: _parse_loop_structure() — UOps → LoopLevel tree
    ├─ Pass 2: _build_datapath()    — combinational signal map
    └─ Pass 3: _build_fsm()         — FSM sequencing stores
```

## Pass 0 — Memories (`hdl_module.py:88`)

Every `DEFINE_GLOBAL` buffer in the UOps becomes an Amaranth `Memory`:

- **Depth**: element count from `dtype.size`
- **Width**: 8 bits for INT8 buffers, 32 bits for INT32
- **Ports**: two combinational read ports (one internal for the datapath, one external for inspection via `buf_read_ports`), one synchronous write port shared between external loading and FSM writes

Default write port wiring routes `buf{N}_wen/waddr/wdata` inputs to the internal write port. FSM states override this with `m.d.comb` when they need to write output.

## Pass 1 — Loop structure (`hdl_module.py:152`)

`_parse_loop_structure()` walks the UOp list and builds a `LoopLevel` tree:

```python
@dataclass
class LoopLevel:
    axis_type: AxisType | None  # LOOP, REDUCE, or None (root)
    bound: int                  # iteration count
    counter: UOp | None         # the RANGE uop
    prologue: list[UOp]         # ops before inner RANGE (or all ops if innermost)
    body: LoopLevel | None      # nested level
    epilogue: list[UOp]         # ops after inner END
```

The root is always returned (axis_type=None). Its body is the outermost RANGE, or None for scalar kernels with no loops.

**Why:** the tree separates *what* to compute from *when* to compute it. Prologue ops (e.g. accumulator reset) belong to the outer level; body ops (MAC) belong to the inner level; epilogue ops (output write) belong back to the outer level.

**LOOP vs REDUCE:** tinygrad tags each RANGE with an `AxisType`. `LOOP` axes are independent iterations (one output element per iteration — parallelizable in principle). `REDUCE` axes are dependent — each iteration reads and updates the accumulator. In hardware today both execute sequentially; the distinction is preserved for future parallelism.

## Pass 2 — Combinational datapath (`hdl_module.py:190`)

`_build_datapath()` walks all UOps and builds a `sig` dict mapping `id(uop) → Signal/value`:

| UOp | Signal |
|-----|--------|
| `DEFINE_GLOBAL` | `None` (handled by memories) |
| `DEFINE_REG` | `acc` Signal (signed 32-bit accumulator) |
| `CONST` | Python int (used directly in `eq()`) |
| `RANGE` | counter Signal from `counter_map` |
| `AFTER` | pass through `sig[src[0]]` (ordering barrier, no new value) |
| `INDEX` into reg | `acc` |
| `INDEX` into buffer | `("index", buf_idx, addr_sig)` — comb address wired to read port |
| `LOAD` from memory | new Signal wired to read port `.data` |
| `MUL/ADD/CAST/CMPLT/WHERE/MAX` | new Signal wired combinationally |
| `STORE` | `None` — resolved at FSM build time |

The datapath is always live. Signals for loop addresses always reflect the current counter values, so when the FSM advances a counter, addresses and loaded data update automatically the next cycle.

## Pass 3 — FSM (`hdl_module.py:280`)

`_build_fsm()` flattens the LoopLevel tree to a list of `(level, depth)` pairs and creates states:

```
IDLE
  └─ on start: reset outermost counter → first state

L{d}_PRO  (non-innermost levels)
  └─ emit prologue STOREs (e.g. acc = 0)
     reset child counter → child's first state

L{d}_BODY  (innermost level)
  └─ emit body STOREs (e.g. acc += MAC result)
     if counter == bound-1: → parent EPI (or set done, → IDLE)
     else: counter++, → self

L{d}_EPI  (non-innermost levels)
  └─ emit epilogue STOREs (e.g. output[i] = cast(acc))
     if counter == bound-1: → parent EPI (or set done, → IDLE)
     else: counter++, → L{d}_PRO
```

**Store resolution** (`_emit_stores`, `hdl_module.py:345`): each `STORE` UOp is resolved at emit time by looking up `sig[id(store.src[0])]`:
- If it's `acc` → `m.d.sync += acc.eq(value)` (register write)
- If it's `("index", buf_idx, addr_sig)` → `m.d.comb += [wp.addr.eq(addr), wp.data.eq(value), wp.en.eq(1)]` (memory write, overrides default external wiring)

**Done signal:** set synchronously (`m.d.sync += done.eq(1)`) in the last compute state when the outermost counter reaches its bound, then cleared in IDLE. This means done is valid for exactly one cycle after the final store.

## Timing model

| Pattern | States | Cycle count |
|---------|--------|-------------|
| Scalar (no RANGE) | IDLE → SCALAR → IDLE | 1 |
| Elementwise (single LOOP, bound N) | IDLE → L0_BODY × N → IDLE | N |
| GEMV (LOOP M, REDUCE K) | IDLE → (L0_PRO + L1_BODY×K + L0_EPI) × M → IDLE | M×(K+2) |

## Reference implementation (`hdl/gemv.py`)

The manual `GEMVUnit` predates the compiler and uses a fixed FSM (IDLE → COMPUTE → EMIT → DONE) hardcoded for the GEMV pattern. Its cycle count is M×(K+1)+1. The compiler generates a comparable structure generically from UOps.

## Overflow analysis

$$\max(\text{acc}) = K \times 127^2 = K \times 16{,}129$$

For K=784: 12,645,136 < 2³¹−1 ✓