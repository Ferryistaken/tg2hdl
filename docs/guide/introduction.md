# Introduction

## Project Overview

`tg2hdl` implements a hardware acceleration framework for neural network inference, mapping computational kernels from the tinygrad deep learning framework to FPGA-targeted hardware descriptions in Amaranth HDL.

### Scope

The current implementation focuses on a minimal viable system:

- **Target model**: 2-layer MLP for MNIST digit classification
- **Core primitive**: INT8 General Matrix-Vector (GEMV) multiplication
- **Architecture**: Sequential single-MAC datapath with FSM control
- **Verification**: Bit-accurate simulation against NumPy reference

## System Workflow

```
tinygrad (model/IR) → kernel inspection → Amaranth HDL (hardware) → simulation verification
```

### Stage 1: Kernel Analysis

The `inspect_kernels.py` script analyzes tinygrad model compilation to identify kernel shapes and computational patterns. For the MNIST MLP:

```python
h = (x @ w1 + b1).relu()      # Kernel 0: (1×784) @ (784×128) → (1×128)
logits = h @ w2 + b2           # Kernel 1: (1×128) @ (128×10) → (1×10)
```

With batch size 1, these reduce to matrix-vector products, motivating the GEMV primitive.

### Stage 2: Hardware Implementation

Amaranth HDL modules implement the computational primitives:

- `hdl/gemv.py`: Sequential GEMV unit with INT8×INT8→INT32 MAC
- `hdl/relu.py`: Combinational ReLU activation

### Stage 3: Verification

Simulation tests (`tests/test_gemv.py`) validate:

- Numerical correctness against NumPy reference
- Cycle-accurate timing behavior
- Edge cases (signed arithmetic, identity matrices, single elements)

## Implementation Status

### Current Capabilities

| Component | Status |
|-----------|--------|
| GEMV unit (single-MAC) | ✅ Implemented |
| INT8×INT8→INT32 MAC | ✅ Implemented |
| FSM control (IDLE/COMPUTE/EMIT/DONE) | ✅ Implemented |
| ReLU activation | ✅ Implemented |
| Simulation testbench | ✅ Implemented |
| Cycle-count validation | ✅ Implemented |

### Planned Extensions

| Feature | Status |
|---------|--------|
| Multi-MAC parallelization | Planned |
| Bias addition chaining | Planned |
| Auto-generation from tinygrad IR | Planned |
| Full layer execution (GEMV+ReLU+bias) | Planned |

## Design Rationale

### Quantization

INT8 weights and activations reduce memory bandwidth and multiplier complexity versus float32, while INT32 accumulation prevents overflow for realistic kernel dimensions (e.g., 784 terms × 127² ≈ 12.6M < 2³¹).

### Sequential Baseline

A single-MAC design establishes correctness and timing behavior before optimization. Parallelization (NUM_MACS > 1) is a straightforward extension once the baseline is validated.

### Python-First Workflow

Both tinygrad and Amaranth use Python, enabling:

- Shared type systems and numerical validation
- Direct NumPy comparison in tests
- Potential for IR-to-HDL compilation (future work)

## Documentation Structure

- **Guides**: Hand-written narrative documentation
- **API Reference**: Auto-generated from Python docstrings via Sphinx autodoc
- **Architecture**: Deep technical specification (see `ARCHITECTURE.md`)