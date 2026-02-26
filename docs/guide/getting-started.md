# Getting Started

## Start here

If you want the project-level context first, read [Introduction](/guide/introduction).

## Repository map

- `ARCHITECTURE.md` – deep technical walkthrough from model math to FSM behavior.
- `hdl/gemv.py` – sequential INT8 GEMV hardware block.
- `hdl/relu.py` – combinational ReLU block.
- `tests/test_gemv.py` – simulation tests and cycle-count checks.
- `inspect_kernels.py` – tinygrad kernel inspection workflow.
- `train_mnist.py` – training script that exports MNIST weights.

## Quickstart

```bash
# Python environment and tests
uv sync
uv run pytest tests/test_gemv.py -k "not slow"

# docs site
cd docs
npm install
npm run docs:dev
```

Then open `http://localhost:4173`.

## Current scope

The current implementation is a correctness-first prototype:

- hand-authored Amaranth GEMV module,
- simulation-based verification,
- and documented path toward automatic generation from tinygrad kernel IR.
