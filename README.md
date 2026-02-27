# tg2hdl

Hardware acceleration framework for neural network inference using Amaranth HDL.

## Overview

This project implements an INT8 GEMV (General Matrix-Vector) unit in Amaranth HDL, designed for executing quantized neural network layers on FPGA targets. The current implementation targets a 2-layer MNIST MLP, with extensibility planned for broader model support.

## Quickstart

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/test_gemv.py -k "not slow"

# Build documentation
uv run sphinx-build -b dirhtml docs docs/_build/dirhtml
```

## Architecture Summary

The system implements a sequential multiply-accumulate GEMV unit with:

- **Data types**: INT8 weights/activations, INT32 accumulation
- **Operation**: y[i] = Σⱼ W[i][j] × x[j]
- **FSM states**: IDLE → COMPUTE → EMIT → DONE
- **Cycle complexity**: O(M × K) for single-MAC configuration

## Verification

Simulation-based verification using Amaranth's bit-accurate simulator with NumPy reference validation. See [Verification](docs/guide/verification.md).

## Documentation

- [Getting Started](docs/guide/getting-started.md)
- [Architecture](docs/guide/architecture.md)
- [API Reference](docs/guide/api-reference.md)
- [Deployment](docs/guide/deployment.md)

## Research Notes

- Current implementation is hand-authored HDL; auto-generation from tinygrad IR is planned
- Single-MAC baseline established; parallelization is the next optimization target
- Cycle-accurate timing model validated through simulation