# System Architecture

## Computational Model

### GEMV Operation

The General Matrix-Vector multiplication computes:

```
y[i] = Σⱼ₌₀^K-1 W[i][j] × x[j]    for i = 0, ..., M-1
```

where:
- **W** ∈ ℤ^(M×K): weight matrix (INT8)
- **x** ∈ ℤ^K: input vector (INT8)
- **y** ∈ ℤ^M: output vector (INT32)

### Target Network

The MNIST MLP consists of two linear layers:

```
Layer 1: h = ReLU(x @ W₁ᵀ + b₁)    # (784) → (128)
Layer 2: y = h @ W₂ᵀ + b₂          # (128) → (10)
```

With batch size 1, matrix multiplications reduce to GEMV operations.

## Hardware Architecture

### GEMV Unit (`hdl/gemv.py`)

```
┌─────────────────────────────────────────────────────────────┐
│                      GEMVUnit                               │
│                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │  vec_mem     │────►│              │     │             │ │
│  │  [K] INT8    │     │  MUL 8×8→16  │────►│  ACC INT32  │ │
│  └──────────────┘     │              │     │  += product │ │
│                       └──────┬───────┘     └─────────────┘ │
│                              │              ▲              │
│  ┌──────────────┐           │              │              │
│  │  w_mem       │───────────┘              │              │
│  │  [M×K] INT8  │                          │              │
│  └──────────────┘                          │              │
│         ▲                                  │              │
│         │         ┌──────────────────────┐ │              │
│         └────────►│      FSM Control     │◄──────────────┘
│                   │  IDLE/COMPUTE/EMIT/  │
│                   │       DONE           │
│                   └──────────────────────┘
│                              │
│         ┌────────────────────┼────────────────────┐
│         │                    │                    │
│  result_idx            result_data         result_valid
│  (output)              (INT32)             (output)
└─────────────────────────────────────────────────────────────┘
```

### Memory Organization

| Memory | Size | Width | Purpose |
|--------|------|-------|---------|
| `vec_mem` | K | 8-bit signed | Input vector storage |
| `w_mem` | M×K | 8-bit signed | Weight matrix (row-major) |

### Interface Signals

#### Control
| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `start` | Input | 1 | Initiate computation |
| `done` | Output | 1 | Computation complete |
| `busy` | Output | 1 | Unit is active |

#### Vector Load Port
| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `vec_wen` | Input | 1 | Write enable |
| `vec_waddr` | Input | ⌈log₂K⌉ | Write address |
| `vec_wdata` | Input | 8 | Write data (INT8) |

#### Weight Load Port
| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `w_wen` | Input | 1 | Write enable |
| `w_waddr` | Input | ⌈log₂(MK)⌉ | Write address |
| `w_wdata` | Input | 8 | Write data (INT8) |

#### Result Output
| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `result_valid` | Output | 1 | Data valid indicator |
| `result_idx` | Output | ⌈log₂M⌉ | Output row index |
| `result_data` | Output | 32 | Computed value (INT32) |

## Finite State Machine

### State Diagram

```
     start=1
 ┌───┐       ┌─────────┐      col==K-1       ┌──────┐      row==M-1     ┌──────┐
 │IDLE│─────►│ COMPUTE │────────────────────►│ EMIT │─────────────────►│ DONE │
 └───┘       └────┬────┘                     └──┬───┘                  └──┬───┘
     ▲            │ │                           │                          │
     │            │ └── col++ (loop K times)    └── not last: row++, reset acc
     │            │                              (transition to COMPUTE)   │
     └────────────┴────────────────────────────────────────────────────────┘
                (return to IDLE)
```

### State Descriptions

#### IDLE
- **Entry conditions**: Initial state or after DONE
- **Actions**: Wait for `start` signal; initialize counters
- **Exit condition**: `start` asserted
- **Duration**: Variable (external control)

#### COMPUTE
- **Entry conditions**: From IDLE (row=0, col=0, acc=0) or EMIT (col=0, acc=0)
- **Actions per cycle**:
  1. Read `vec_mem[col_idx]` and `w_mem[row_idx×K + col_idx]`
  2. Compute product = W[row][col] × x[col] (combinational)
  3. Accumulate: acc ← acc + product (synchronous)
  4. Increment `col_idx`
- **Exit condition**: `col_idx` == K-1
- **Duration**: K cycles per row

#### EMIT
- **Entry conditions**: From COMPUTE (col==K-1)
- **Actions**:
  1. Assert `result_valid`
  2. Drive `result_data` = acc, `result_idx` = row_idx
  3. Reset acc = 0, col_idx = 0
  4. Increment `row_idx` (if not last row)
- **Exit condition**: Always transitions after 1 cycle
- **Duration**: 1 cycle per row

#### DONE
- **Entry conditions**: From EMIT (row==M-1)
- **Actions**: Assert `done`
- **Exit condition**: Always transitions to IDLE after 1 cycle
- **Duration**: 1 cycle

## Timing Analysis

### Cycle Complexity

For dimensions M×K with single MAC:

```
T_cycles = M × (K + 1) + 1
         = M×K (compute) + M (emit) + 1 (done)
```

### Latency Estimates

At frequency f (MHz):

```
T_latency (μs) = T_cycles / f
```

| Kernel | M | K | Cycles | @25 MHz | @100 MHz | @200 MHz | @300 MHz |
|--------|---|---|--------|---------|----------|----------|----------|
| Layer 1 | 128 | 784 | 100,480 | 4.02 ms | 1.00 ms | 0.50 ms | 0.33 ms |
| Layer 2 | 10 | 128 | 1,290 | 51.6 μs | 12.9 μs | 6.45 μs | 4.30 μs |
| **Total** | - | - | **101,770** | **4.07 ms** | **1.02 ms** | **0.51 ms** | **0.34 ms** |

### Parallelization Scaling

With N MACs operating in parallel:

```
T_cycles(N) = M × (⌈K/N⌉ + 1) + 1
```

| MACs | Layer 1 Cycles | Layer 2 Cycles | Total @200 MHz |
|------|----------------|----------------|----------------|
| 1 | 100,352 | 1,280 | 0.51 ms |
| 8 | 12,592 | 166 | 63.8 μs |
| 32 | 3,184 | 50 | 16.2 μs |
| 64 | 1,600 | 40 | 8.2 μs |
| 128 | 832 | 20 | 4.3 μs |

## ReLU Unit (`hdl/relu.py`)

### Combinational Implementation

```
┌─────────────┐
│    ReLU     │
│             │
│  inp ──────►│ max(0, inp) ────► out
│  (INT32)    │     (INT32)      (INT32)
└─────────────┘
```

### Logic

```verilog
// Pseudocode representation
out = (inp < 0) ? 0 : inp;
```

- **Latency**: 0 cycles (combinational)
- **Critical path**: Sign bit check + mux

## Resource Estimation

### Per-MAC Resources (Xilinx 7-series)

| Component | Resource | Quantity |
|-----------|----------|----------|
| Multiplier | DSP48 | 1 (INT8×INT8→INT16) |
| Adder | LUTs | ~10 (INT16+INT32→INT32) |
| Accumulator | Flip-flops | 32 |

### Memory Resources

| Memory | Depth | Width | BRAM (36Kb) |
|--------|-------|-------|-------------|
| vec_mem | 784 | 8-bit | ~0.2 KB |
| w_mem (L1) | 100,352 | 8-bit | ~98 KB (~3 BRAM) |
| w_mem (L2) | 1,280 | 8-bit | ~1.3 KB |

**Total BRAM**: ~4 blocks (36Kb each)

### FSM Resources

| Component | LUTs | FFs |
|-----------|------|-----|
| State encoding | 2 | 2 |
| Row counter | ~8 | 8 |
| Col counter | ~10 | 10 |
| Control logic | ~20 | - |

**Total**: ~40 LUTs, 20 FFs (negligible vs datapath)

## Design Constraints

### Overflow Prevention

Maximum accumulation value for INT8 inputs:

```
max(acc) = K × 127 × 127 = K × 16,129

For K = 784:  max(acc) = 12,645,136 < 2³¹ - 1 = 2,147,483,647
```

INT32 accumulator provides sufficient headroom.

### Memory Bandwidth

Per-cycle bandwidth requirement:

- vec_mem: 1 read/cycle (8-bit)
- w_mem: 1 read/cycle (8-bit)

Both well within FPGA block RAM capabilities.

### Critical Path

```
Memory read → MUL8x8 → ADD32 → Register
```

Typical timing:
- @25 MHz (iCE40): Easily met
- @100 MHz (ECP5): Comfortable
- @200 MHz (Artix-7): Requires careful placement
- @300 MHz (Kintex-7): May need pipeline staging