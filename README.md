# tg2hdl

Neural network inference accelerators for FPGA deployment.

## Overview

Hardware implementation of INT8 GEMV (matrix-vector multiplication) primitives targeting MNIST MLP inference. Uses Amaranth HDL for synthesizable circuit description with bit-accurate NumPy validation.

## Architecture

**GEMV Unit** (`hdl/gemv.py`)
- INT8×INT8→INT32 multiply-accumulate
- FSM control: IDLE → COMPUTE → EMIT → DONE
- Cycle complexity: O(M×K)

**Target Network**
- Layer 1: 784→128 (ReLU)
- Layer 2: 128→10
- Total MACs: 101,632

## Quick Start

```bash
# Tests
uv run pytest

# Benchmarks
uv run python benchmark.py
```

## Performance

| Platform | Latency | Efficiency |
|----------|---------|------------|
| CPU (NumPy) | 9.6 μs | 1.42 GOPS/W |
| FPGA 64-MAC @200MHz | 9.1 μs | 14.87 GOPS/W |

FPGA achieves 10× better energy efficiency.

## Documentation

See [`docs/`](docs/index.rst) for detailed architecture and verification.