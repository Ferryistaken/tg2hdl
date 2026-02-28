# Introduction

## Overview

`tg2hdl` implements INT8 GEMV primitives for FPGA-based neural network inference, targeting the MNIST MLP model from tinygrad.

## Components

- **`hdl/gemv.py`**: Sequential GEMV unit (INT8×INT8→INT32)
- **`hdl/relu.py`**: Combinational ReLU activation
- **`tests/test_gemv.py`**: Bit-accurate simulation tests

## Workflow

```
tinygrad model → kernel inspection → Amaranth HDL → simulation verification
```

## Status

| Component | Status |
|-----------|--------|
| GEMV unit (single-MAC) | ✅ |
| ReLU activation | ✅ |
| Simulation testbench | ✅ |
| Multi-MAC parallelization | Planned |
| Compiler backend | WIP |