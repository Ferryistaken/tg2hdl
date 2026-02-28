# Benchmarks

**Model:** MNIST MLP, 2 layers (784→128 ReLU, 128→10), 101,632 total MACs.

## Compiler-generated hardware (single-MAC, simulated)

The cycle count formula for a compiled GEMV kernel is M×(K+2) — one cycle for acc reset, K cycles for MACs, one cycle for output write — repeated M times.

| Kernel | Dimensions | Cycles | @100 MHz |
|--------|------------|--------|----------|
| Layer 1 | 128×784 | 101,504 | ~1.0 ms |
| Layer 2 | 10×128 | 1,300 | ~13 μs |
| **Total** | | **~102,800** | **~1.0 ms** |

## Parallelism potential (not yet implemented)

Enabling tinygrad's `UNROLL` optimization would expose N-wide SIMD in the UOps, allowing N parallel MACs. The cycle count scales as M×(⌈K/N⌉+2).

| MACs | Layer 1 cycles | Layer 2 cycles | Total @200 MHz |
|------|----------------|----------------|----------------|
| 1 | 101,504 | 1,300 | ~0.51 ms |
| 8 | 12,928 | 170 | ~65 μs |
| 64 | 1,664 | 30 | ~8.5 μs |
| 128 | 896 | 20 | ~4.6 μs |

## End-to-end comparison

```bash
uv run python compare_inference.py
```

Runs a single MNIST test image through tinygrad float32 (CPU reference) and through the two compiled kernels (Amaranth simulation, INT8 quantized). Prints predictions, cycle counts, and wall-clock simulation time.