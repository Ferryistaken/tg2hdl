# Verification

## Test Infrastructure

Verification uses Amaranth's cycle-accurate simulator with NumPy as ground truth reference.

### Simulation Framework

```python
# Test pattern
dut = GEMVUnit(m_dim=M, k_dim=K)
sim = Simulator(dut)
sim.add_clock(1e-8)  # 100 MHz

# 1. Load inputs (vec_mem, w_mem)
# 2. Assert start signal
# 3. Collect outputs on result_valid
# 4. Compare against NumPy reference
# 5. Assert done signal
```

## Test Coverage

### Numerical Correctness

| Test | Dimensions | Purpose |
|------|------------|---------|
| `test_2x2` | 2×2 | Minimal case, manual verification |
| `test_4x3` | 4×3 | Non-square matrix |
| `test_negative_values` | 2×2 | Signed INT8 arithmetic |
| `test_identity` | 3×3 | Identity matrix preserves input |
| `test_single_element` | 1×1 | Edge case |
| `test_random_8x16` | 8×16 | Random INT8 validation |
| `test_kernel1_10x128` | 10×128 | Actual MNIST kernel |

### Timing Validation

| Test | Assertion |
|------|-----------|
| `test_cycle_count_4x3` | Cycles = M×(K+1) |

### Activation Functions

| Test | Assertion |
|------|-----------|
| `test_relu_positive` | x > 0 → output = x |
| `test_relu_negative` | x < 0 → output = 0 |
| `test_relu_zero` | x = 0 → output = 0 |

## Execution

```bash
# Full suite
uv run pytest

# Fast iteration (exclude MNIST-sized tests)
uv run pytest tests/test_gemv.py -k "not slow"

# With VCD waveform output
uv run pytest -s tests/test_gemv.py::test_2x2
```

## Waveform Debugging

Tests generate `gemv_test.vcd` for cycle-by-cycle inspection in Verilator GTKWave or similar tools.