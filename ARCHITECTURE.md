# How This Works: From Neural Network to Hardware

## The Big Picture

We have a tiny neural network that recognizes handwritten digits (MNIST). In `inspect_kernels.py`, tinygrad compiles it into a sequence of **kernels** — low-level operations the computer actually runs. The two that matter are:

```python
h = (x @ w1 + b1).relu()   # Kernel 0: multiply 1×784 by 784×128, add bias, ReLU
logits = h @ w2 + b2        # Kernel 1: multiply 1×128 by 128×10, add bias
```

The `@` operator is matrix multiplication. But since our input `x` is a single vector (1 row), these are really **matrix-vector multiplies** — that's what GEMV means.

Our goal: build this operation as a hardware circuit that could run on an FPGA, described in Python using the Amaranth HDL library.

## What is a GEMV?

GEMV = **GE**neral **M**atrix-**V**ector multiply.

Given a weight matrix W (M rows × K columns) and an input vector x (K elements), compute:

```
y[0] = W[0][0]*x[0] + W[0][1]*x[1] + ... + W[0][K-1]*x[K-1]
y[1] = W[1][0]*x[0] + W[1][1]*x[1] + ... + W[1][K-1]*x[K-1]
...
y[M-1] = W[M-1][0]*x[0] + W[M-1][1]*x[1] + ... + W[M-1][K-1]*x[K-1]
```

Each output element `y[i]` is the **dot product** of row `i` of the weight matrix with the input vector. That's it — it's just multiply-and-add, repeated many times.

For our MNIST network:
- Kernel 0: M=128, K=784 → 128 dot products, each summing 784 terms = 100,352 multiplications
- Kernel 1: M=10, K=128 → 10 dot products, each summing 128 terms = 1,280 multiplications

## What is a MAC?

MAC = **M**ultiply-**AC**cumulate. It's the atomic operation inside a dot product:

```
accumulator = accumulator + (a * b)
```

One MAC per clock cycle. To compute one dot product of length K, you need K MAC operations. Our hardware does them one at a time (sequentially), so one output element takes K clock cycles.

In hardware, this is literally:
1. Read `W[i][j]` and `x[j]` from memory
2. Multiply them (8-bit × 8-bit → 16-bit result)
3. Add the product to a running 32-bit accumulator
4. Increment j, repeat

## Why INT8 and INT32?

- **INT8 inputs**: Weights and activations are 8-bit signed integers (-128 to +127). This is "quantized" inference — real neural networks often use float32, but INT8 is much cheaper in hardware (smaller multipliers, less memory).
- **INT32 accumulator**: When you multiply two INT8 values, the result can be up to 16 bits. When you sum K of these products, the result can overflow 16 bits. Worst case for K=784: 784 × 127 × 127 = 12,636,688, which fits in 32 bits (max ~2 billion). So 32-bit accumulation guarantees no overflow.

## The Hardware Design (hdl/gemv.py)

### What Amaranth Does

Amaranth is a Python library for describing digital circuits. Instead of writing Verilog or VHDL, you write Python that **describes** what the hardware looks like. The Python doesn't run on the FPGA — it generates the circuit description that gets synthesized into actual gates and flip-flops.

Key concepts:
- **Signal**: A wire or register in the circuit. `Signal(signed(8))` is an 8-bit signed wire.
- **m.d.comb +=**: Combinational logic — output changes immediately when input changes (like wiring).
- **m.d.sync +=**: Sequential logic — output updates on the next clock edge (like a register/flip-flop).
- **Memory**: An array of values accessible by address, like SRAM on a chip.
- **FSM**: Finite state machine — the circuit steps through states on each clock cycle.

### The Circuit

```
                 ┌─────────────────────────────────────────────┐
                 │                GEMVUnit                     │
                 │                                             │
  vec_wdata ────►│  vec_mem[K]     ┌──────┐    ┌───────────┐   │
  vec_waddr ────►│  (INT8 array)──►│ MUL  │───►│ ACC (32b) │   │
  vec_wen   ────►│       ▲         │ 8×8  │    │ += prod   │   │
                 │       │         └──────┘    └─────┬─────┘   │
  w_wdata  ─────►│  w_mem[M×K]        ▲              │         │
  w_waddr  ─────►│  (INT8 array)──────┘              ▼         │
  w_wen    ─────►│       ▲                     result_data ────►│
                 │       │                     result_idx  ─────►│
  start    ─────►│   FSM: steps through        result_valid ────►│
  done     ◄────│   rows & columns             busy ───────────►│
                 └─────────────────────────────────────────────┘
```

### The State Machine

The FSM controls what happens each clock cycle:

```
   ┌──────┐  start=1   ┌─────────┐  col==K-1   ┌──────┐  last row   ┌──────┐
   │ IDLE ├───────────►│ COMPUTE ├────────────►│ EMIT ├───────────►│ DONE │
   └──┬───┘            └────┬────┘             └──┬───┘            └──┬───┘
      ▲                     │                     │                    │
      │                     └─── col++ ◄──────────┘ not last row       │
      │                     (loop K times)        (next row)           │
      └────────────────────────────────────────────────────────────────┘
```

**IDLE**: Waiting. When `start` goes high, initialize counters and go to COMPUTE.

**COMPUTE**: The main loop. Each clock cycle:
1. Set memory addresses: `vec_mem[col_idx]` and `w_mem[row_idx * K + col_idx]`
2. The multiplier computes `product = W[row][col] * x[col]` (combinational — instant)
3. On the next clock edge: `acc <= acc + product` (registered — takes one cycle)
4. Advance `col_idx`. After K cycles, go to EMIT.

**EMIT**: One cycle to output the result. The accumulator holds the complete dot product for this row. Output it on `result_data`, flag `result_valid`, and report `result_idx`. Then either start the next row (reset acc, increment row) or go to DONE if all rows are finished.

**DONE**: Signal completion for one cycle, return to IDLE.

### Cycle-by-Cycle Example: 2×2 GEMV

W = [[1, 2], [3, 4]], x = [5, 6]

Expected: y[0] = 1×5 + 2×6 = 17, y[1] = 3×5 + 4×6 = 39

```
Cycle  State    row  col  addr_w  addr_x  product  acc (after edge)  output
─────  ───────  ───  ───  ──────  ──────  ───────  ────────────────  ──────
  0    IDLE      -    -     -       -        -      → 0               start=1
  1    COMPUTE   0    0     0       0       1×5=5   → 0+5 = 5
  2    COMPUTE   0    1     1       1       2×6=12  → 5+12 = 17      col==K-1 → EMIT
  3    EMIT      0    -     -       -        -      → 0 (reset)       result_data=17 ✓
  4    COMPUTE   1    0     2       0       3×5=15  → 0+15 = 15
  5    COMPUTE   1    1     3       1       4×6=24  → 15+24 = 39     col==K-1 → EMIT
  6    EMIT      1    -     -       -        -      -                 result_data=39 ✓
  7    DONE      -    -     -       -        -      -                 done=1
```

Total compute cycles: M × (K + 1) = 2 × (2 + 1) = 6 cycles (not counting IDLE/DONE transitions).

## The Tests (tests/test_gemv.py)

### How Amaranth Simulation Works

Amaranth doesn't synthesize to an FPGA for testing. Instead, it **simulates** the circuit in Python, cycle by cycle. The simulator:

1. Creates a virtual clock (`sim.add_clock(1e-8)` = 100 MHz)
2. Runs a Python `async` function as the testbench
3. The testbench pokes input signals (`ctx.set`) and reads output signals (`ctx.get`)
4. `await ctx.tick()` advances one clock cycle

It's like having the FPGA on your desk, but in software. The simulation is **bit-accurate** — it computes exactly what the real hardware would.

### Test Structure

Each test follows this pattern:

```python
# 1. Create the Device Under Test with specific dimensions
dut = GEMVUnit(m_dim=4, k_dim=3)

# 2. Create numpy reference: what the answer SHOULD be
expected = W.astype(np.int32) @ x.astype(np.int32)

# 3. Run simulation:
#    a. Load vector into vec_mem (one element per clock cycle)
#    b. Load weights into w_mem (one element per clock cycle)
#    c. Pulse start signal
#    d. Wait for result_valid, capture results
#    e. Wait for done

# 4. Compare hardware output to numpy reference
assert results[i] == expected[i]
```

### What Each Test Covers

| Test | Size | Why |
|------|------|-----|
| `test_2x2` | 2×2 | Minimal case, easy to trace by hand |
| `test_4x3` | 4×3 | Non-square, multiple rows |
| `test_negative_values` | 2×2 | Signed arithmetic (negative × negative, etc.) |
| `test_identity` | 3×3 | Identity matrix — output should equal input |
| `test_single_element` | 1×1 | Edge case: just one multiplication |
| `test_random_8x16` | 8×16 | Random INT8 values, checks against numpy |
| `test_cycle_count_4x3` | 4×3 | Verifies exact number of clock cycles |
| `test_kernel1_10x128` | 10×128 | Actual MNIST kernel 1 dimensions |

### The ReLU Tests

ReLU is trivial: `output = max(0, input)`. It's a purely combinational circuit (no clock needed). We test positive (passthrough), negative (clamp to 0), and zero. In the full network, ReLU sits after kernel 0's GEMV.

## Timing Estimates

At 100 MHz (10 ns per cycle):

| Kernel | Dimensions | Compute Cycles | Wall Time |
|--------|-----------|---------------|-----------|
| Kernel 0 | 128 × 784 | 128 × 785 = 100,480 | ~1.0 ms |
| Kernel 1 | 10 × 128 | 10 × 129 = 1,290 | ~12.9 µs |
| **Total** | | | **~1.0 ms** |

This is for a single-MAC design (one multiply per cycle). Real FPGA designs use multiple MACs in parallel (NUM_MACS > 1) to go faster — that's a future optimization.

## What's Next

This is the "can we do it at all?" prototype. Future steps:
- Parallelize with multiple MACs (NUM_MACS parameter)
- Chain GEMV + ReLU for kernel 0
- Add bias addition
- Handle kernel 0's full 128×784 dimensions efficiently
- Eventually: auto-generate this from tinygrad's kernel IR
