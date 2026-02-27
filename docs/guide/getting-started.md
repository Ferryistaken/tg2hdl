# Getting Started

## Repository Structure

```
tg2hdl/
├── hdl/
│   ├── gemv.py          # INT8 GEMV hardware unit
│   └── relu.py          # ReLU activation unit
├── tests/
│   └── test_gemv.py     # Simulation testbench
├── compiler/            # IR-to-HDL compilation (WIP)
├── docs/
│   └── guide/           # Documentation source
├── benchmark.py         # Performance benchmarks
├── train_mnist.py       # Model training script
├── inspect_kernels.py   # Kernel analysis utility
└── ARCHITECTURE.md      # Technical specification
```

## Installation

```bash
# Install dependencies
uv sync
```

## Running Tests

```bash
# Full test suite
uv run pytest

# Exclude slow tests (MNIST-sized kernels)
uv run pytest tests/test_gemv.py -k "not slow"
```

## Building Documentation

```bash
# Build Sphinx documentation
uv run sphinx-build -b dirhtml docs docs/_build/dirhtml

# Serve locally
python -m http.server 4173 -d docs/_build/dirhtml
```

Open `http://localhost:4173`.

## Running Benchmarks

```bash
# CPU vs FPGA vs GPU comparison
uv run python benchmark.py
```

## Quick Validation

```bash
# Train MNIST model and export weights
uv run python train_mnist.py

# Inspect kernel shapes
uv run python inspect_kernels.py
```