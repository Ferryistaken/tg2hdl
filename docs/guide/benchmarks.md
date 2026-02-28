# Benchmarks

**Model:** MNIST MLP (101,632 MACs)

## Key Results

| Platform | Latency | Efficiency |
|----------|---------|------------|
| CPU (NumPy) | 9.6 μs | 1.42 GOPS/W |
| FPGA 64-MAC @200MHz | 9.1 μs | 14.87 GOPS/W |

FPGA achieves 10× better energy efficiency than CPU.

## Cycle Model

$$T_{\text{cycles}}(N) = M \times (\lceil K/N \rceil + 1) + 1$$

| MACs | Cycles | @200 MHz |
|------|--------|----------|
| 1 | 101,770 | 0.51 ms |
| 8 | 12,842 | 64.2 μs |
| 64 | 1,822 | 9.1 μs |
| 128 | 1,044 | 5.2 μs |

## Execution

```bash
uv run python benchmark.py
```