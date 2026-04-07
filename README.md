# tg2hdl

A compiler from tinygrad's IR to synthesizable FPGA hardware.

## What it does

You write a neural network in tinygrad. tg2hdl compiles it to an Amaranth HDL module — memories for buffers, an FSM for loop control, and combinational logic for arithmetic. The result simulates cycle-accurately and is ready for synthesis.

```python
from compiler import HDLRenderer, compile_kernel, simulate_kernel
from compiler.backend import _get_uops

renderer = HDLRenderer()
x = Tensor.empty(1, 3, dtype=dtypes.int8)
w = Tensor.empty(3, 4, dtype=dtypes.int8)
out = (x @ w).cast(dtypes.int32)

uops = _get_uops(out.schedule()[0].ast, renderer)
kernel = compile_kernel(uops)
output, cycles, wall = simulate_kernel(kernel, {1: x_data, 2: w_data})
```

Multi-kernel models are compiled as a dependency DAG — the compiler detects producer/consumer relationships between kernels, topologically sorts them, and generates an FSM that copies data along every edge (including non-adjacent skip connections and fan-out broadcasts). A residual MLP compiles and simulates correctly today.

## Quick Start

```bash
uv run pytest                      # 169+ tests
uv run python compare_inference.py # CPU vs HDL MNIST comparison
```

## Documentation

See [`docs/`](docs/index.rst) for architecture, verification, and benchmarks.