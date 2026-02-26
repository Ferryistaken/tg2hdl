# Introduction

## Project goal

`tg2hdl` explores a practical question: **can we take neural-network math from tinygrad and map it to a hardware-friendly datapath?**

The current focus is intentionally narrow and concrete:

- a tiny 2-layer MNIST MLP,
- the key matrix-vector kernels in that model,
- and a sequential INT8 GEMV implementation in Amaranth HDL.

This repo is a prototype to validate correctness, interfaces, and timing behavior before bigger steps like kernel auto-generation and parallel MAC architectures.

## End-to-end flow today

At a high level, the workflow is:

1. Build / inspect a tinygrad model and identify kernel shapes (`inspect_kernels.py`).
2. Train and export model parameters (`train_mnist.py` → `mnist_weights.safetensors`).
3. Implement hardware building blocks in Amaranth (`hdl/gemv.py`, `hdl/relu.py`).
4. Validate behavior cycle-by-cycle in simulation (`tests/test_gemv.py`).

The current hardware primitive is GEMV because the model is run with batch size 1, so matrix multiplications collapse to matrix-vector products.

## Is this auto-generated from Python today?

**Partially, but not end-to-end auto-generated yet.**

- tinygrad is used in Python to define/inspect model compute and kernel structure.
- Amaranth is also Python, but the HDL module (`GEMVUnit`) is currently **hand-authored**.
- Tests compare the hardware simulation output against NumPy reference math for correctness.

So the project already has Python at every stage, but there is **not yet** an automatic compiler pass that converts tinygrad kernel IR directly into Amaranth modules. That is a planned direction.

## What exists now vs. what is planned

### Implemented now

- Sequential single-MAC GEMV with INT8 × INT8 multiply and INT32 accumulation.
- FSM-driven control (`IDLE → COMPUTE → EMIT → DONE`).
- Deterministic and randomized simulation tests, including timing checks.

### Planned next

- Multi-MAC parallelization.
- Bias and activation chaining for fuller layer execution.
- Better handling of larger kernel dimensions.
- **Auto-generation from tinygrad kernel IR to hardware templates**.

## Why this shape is valuable

This incremental structure keeps risk low:

- validate numeric behavior early,
- make cycle costs explicit,
- and keep hardware/software assumptions transparent.

In short, this repo is the bridge between model-kernel understanding and hardware realization, with correctness-first tooling to support iteration.

## Are these docs auto-generated from Python docstrings?

Partially. Guides are hand-written, and API reference pages are auto-generated with **Sphinx autodoc**.

- Narrative pages in `docs/guide/*.md` are written manually.
- `docs/guide/api-reference.md` renders API docs from Python modules/docstrings automatically via `automodule`.
- This gives both high-level explainers and synchronized code-level reference.

