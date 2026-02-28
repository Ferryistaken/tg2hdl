# Verification

## Test Suite

Amaranth cycle-accurate simulation with NumPy ground truth.

## Coverage

| Test | Dimensions | Validates |
|------|------------|-----------|
| `test_2x2` | 2×2 | Minimal case |
| `test_4x3` | 4×3 | Non-square |
| `test_negative_values` | 2×2 | Signed INT8 |
| `test_identity` | 3×3 | Identity matrix |
| `test_single_element` | 1×1 | Edge case |
| `test_random_8x16` | 8×16 | Random INT8 |
| `test_kernel1_10x128` | 10×128 | MNIST kernel |
| `test_cycle_count_4x3` | 4×3 | Timing model |

**ReLU:** `test_relu_positive`, `test_relu_negative`, `test_relu_zero`

## Execution

```bash
uv run pytest                      # Full suite
uv run pytest -k "not slow"        # Fast iteration
```

Tests generate `gemv_test.vcd` for waveform debugging.