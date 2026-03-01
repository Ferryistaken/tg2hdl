# Benchmark Guide

This guide shows how to benchmark any tinygrad computation against the
HDL compiler's simulation.  It covers:

- **Integer (int8/int16/int32)** — full Amaranth simulation, bit-exact results
- **Float32** — full Amaranth simulation with IEEE 754 hardware modules, `rtol=1e-5`
- How to write your own benchmark
- How to interpret the output
- Advanced: multi-kernel chaining, TopModule, quantization workflow

---

## Quick Start

### Run the existing suite

```bash
# Correctness suite: Tier 1 elementwise, Tier 2 GEMV, Tier 3 multi-kernel MLP
uv run pytest benchmarks/test_suite.py -v

# Performance suite: 10 workloads with cycle-count reporting
uv run pytest benchmarks/test_perf_suite.py -v

# Everything except slow tests
uv run pytest tests/ benchmarks/ -k "not slow" -v
```

### Interactive harness

```bash
uv run python -m benchmarks.harness
```

---

## Architecture Overview

```
Your tinygrad graph
        │
        ▼
  Tensor.schedule()          ← produces list[ExecItem] with Buffer objects
        │
        ▼
  compile_model(schedule)    ← produces list[KernelSpec]  (CompiledKernel per op)
        │
        ▼
  simulate_kernel(kernel, inputs)  ← Amaranth Simulator, returns (output, cycles, wall_s)
        │
        ▼
  BenchResult                ← correct?, max_error, hdl_cycles, timing
```

Multi-kernel graphs (e.g., 2-layer MLP) chain `simulate_kernel` calls
automatically by detecting shared Buffer objects between schedule items.

---

## Writing a Custom Benchmark

### Minimal example — single kernel, int32

```python
import numpy as np
from tinygrad import Tensor, dtypes
from benchmarks.harness import run_bench

rng = np.random.RandomState(42)

result = run_bench(
    "my_elementwise_relu",
    lambda tensors: tensors[0].relu(),
    [rng.randint(-10, 10, 64).astype(np.int32)],
)
print(result)
assert result.correct
print(f"Cycles: {result.hdl_cycles}")
```

### Single-kernel GEMV — int8 weights, int32 output

```python
result = run_bench(
    "my_gemv",
    lambda t: (t[0] @ t[1]).cast(dtypes.int32),
    [
        rng.randint(-5, 5, (1, 8)).astype(np.int8),   # input x
        rng.randint(-5, 5, (8, 16)).astype(np.int8),  # weight matrix
    ],
)
print(result)
```

### Linear layer with bias and ReLU

```python
result = run_bench(
    "my_linear_relu",
    lambda t: ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu(),
    [
        rng.randint(-4, 4, (1, 8)).astype(np.int8),
        rng.randint(-4, 4, (8, 16)).astype(np.int8),
        rng.randint(-20, 20, (1, 16)).astype(np.int32),  # bias
    ],
)
```

### Two-layer MLP — multi-kernel

Multi-kernel graphs work exactly the same way.  The harness auto-detects
which buffer connects the two kernels.

```python
def build_mlp(t):
    h = ((t[0] @ t[1]).cast(dtypes.int32) + t[2]).relu()
    return (h.cast(dtypes.int8) @ t[3]).cast(dtypes.int32) + t[4]

result = run_bench(
    "my_2layer_mlp",
    build_mlp,
    [
        rng.randint(-4, 4, (1, 8)).astype(np.int8),   # x
        rng.randint(-3, 3, (8, 4)).astype(np.int8),   # w1
        rng.randint(-10, 10, (1, 4)).astype(np.int32), # b1
        rng.randint(-3, 3, (4, 2)).astype(np.int8),   # w2
        rng.randint(-5, 5, (1, 2)).astype(np.int32),  # b2
    ],
)
```

---

## Float Models

The HDL compiler supports **float32** natively via dedicated IEEE 754 hardware
modules (`FP32Add`, `FP32Mul`, `FP32Cmp`) that produce bit-accurate results in
both Amaranth simulation and synthesis.  Float32 benchmarks use the same
hardware simulation path as integer benchmarks — `result.float_path` is always
`False`.

```
result.float_path == False   (always — no software fallback)
result.correct    == True    (IEEE 754 result matches tinygrad CPU within rtol=1e-5)
result.hdl_cycles > 0        (real hardware cycle count, not analytical)
```

### Example — float32 model

```python
result = run_bench(
    "relu_float32",
    lambda t: t[0].relu(),
    [rng.randn(64).astype(np.float32)],
)
assert not result.float_path   # uses hardware sim, not software fallback
assert result.correct          # IEEE 754 result within rtol=1e-5
print(f"Cycles: {result.hdl_cycles}")
```

### Float32 limitations

- Subnormal numbers flush to zero.
- Rounding is truncation (round-toward-zero), not IEEE default round-to-nearest-even.
- Float16 / BFloat16 arithmetic is **not supported** — these dtypes have no
  dedicated hardware units and will raise `NotImplementedError` at compile time
  for any arithmetic op.

### When you need quantized integer inference

Quantize your model to int8 first — this gives bit-exact simulation and much
better simulation throughput:

```python
from utils import quantize_int8, dequantize

# Load float32 model
w_float = load_weights()   # (out_features, in_features) float32
x_float = load_input()     # (in_features,) float32

# Quantize
w_q, w_scale = quantize_int8(w_float.T)   # transpose to (in, out) for matmul
x_q, x_scale = quantize_int8(x_float)

# Express in tinygrad with int8 dtype
result = run_bench(
    "my_layer_quantized",
    lambda t: (t[0] @ t[1]).cast(dtypes.int32),
    [x_q.reshape(1, -1), w_q],
)
assert result.correct   # bit-exact simulation

# Dequantize output
y_float = dequantize(result.output_hdl, w_scale * x_scale)
```

---

## Interpreting Results

### `result.hdl_cycles`

The number of clock cycles the hardware FSM takes from `start` to `done`.

For a GEMV kernel computing output of shape `(M,)` from input `(K,)`:

```
cycles = M × (K + 2) + 1
```

Example: `(1, 8) @ (8, 16)` gives M=16, K=8 → 16 × 10 + 1 = **161 cycles**.

At 100 MHz this is 1.61 µs.  Scale to your target clock frequency as needed.

### `result.sim_wall_s`

Wall-clock time for the **Amaranth Python simulation**.  This is much slower
than real hardware (seconds vs. nanoseconds).  It scales roughly as O(M×K)
per kernel.  Large kernels (MNIST 784→128) take minutes to simulate.

### `result.correct` and `result.max_abs_error`

For **integer** paths, `correct=True` means the HDL output is **bit-identical**
to tinygrad CPU.  A non-zero `max_abs_error` indicates a logic bug.

For **float32** paths, `correct=True` means the IEEE 754 hardware simulation
result matches tinygrad CPU within `rtol=1e-5, atol=1e-6`.  A small non-zero
`max_abs_error` is normal (rounding differences); a large error indicates a
hardware module bug.

---

## Adding a Pytest Benchmark

```python
# benchmarks/my_benchmarks.py
import os
os.environ.setdefault("NOOPT", "1")
os.environ.setdefault("DEBUG", "0")

import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from benchmarks.harness import run_bench

rng = np.random.RandomState(0)

@pytest.mark.parametrize("M,K", [(4, 4), (8, 8), (16, 16)])
def test_gemv_shapes(M, K):
    x = rng.randint(-4, 4, (1, K)).astype(np.int8)
    w = rng.randint(-4, 4, (K, M)).astype(np.int8)

    r = run_bench(f"gemv_{M}x{K}", lambda t: (t[0] @ t[1]).cast(dtypes.int32), [x, w])
    print(f"\n  {r}")
    assert r.correct
    # Verify expected cycle count
    expected = M * (K + 2) + 1
    assert abs(r.hdl_cycles - expected) <= 2
```

Run it with:

```bash
uv run pytest benchmarks/my_benchmarks.py -v -s
```

---

## Advanced: TopModule Hardware Simulation

For applications requiring verified multi-kernel hardware behavior (not just
chained software simulation), use `TopModule`:

```python
from compiler import compile_top_module
from compiler.top_module import simulate_top

from tinygrad import Tensor, dtypes

x   = Tensor.empty(1, 4, dtype=dtypes.int8)
w1  = Tensor.empty(4, 3, dtype=dtypes.int8)
b1  = Tensor.empty(1, 3, dtype=dtypes.int32)
w2  = Tensor.empty(3, 2, dtype=dtypes.int8)
b2  = Tensor.empty(1, 2, dtype=dtypes.int32)

h      = ((x @ w1).cast(dtypes.int32) + b1).relu()
logits = (h.cast(dtypes.int8) @ w2).cast(dtypes.int32) + b2

schedule = logits.schedule()
top, connections, kernel_specs = compile_top_module(schedule)

# Load inputs via ext_write_ports
import numpy as np
rng = np.random.RandomState(0)
input_data = {
    (0, 1): rng.randint(-4, 4, 4).astype(np.int8),   # k0 buf1 = x
    (0, 2): rng.randint(-3, 3, 12).astype(np.int8),  # k0 buf2 = w1
    (0, 3): rng.randint(-10, 10, 3).astype(np.int32), # k0 buf3 = b1
    (1, 2): rng.randint(-3, 3, 6).astype(np.int8),   # k1 buf2 = w2
    (1, 3): rng.randint(-5, 5, 2).astype(np.int32),  # k1 buf3 = b2
}

output, cycles, wall = simulate_top(top, input_data)
print(f"Output: {output}, Cycles: {cycles}")
```

`TopModule` uses a hardware DMA copy FSM between kernels — the output of
kernel N is copied into kernel N+1's input buffer one element per cycle before
computation begins.

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `NOOPT=1` | Disables tinygrad loop tiling / unrolling — required for correct HDL generation |
| `DEBUG=0` | Suppresses tinygrad debug output |

Always set `NOOPT=1` before building the schedule for HDL compilation.

---

## Project Layout

```
compiler/
  __init__.py        public API
  backend.py         HDLRenderer, compile_kernel, compile_model, compile_top_module,
                     simulate_kernel, count_cycles_from_schedule
  hdl_module.py      CompiledKernel (Amaranth Elaboratable)
  top_module.py      TopModule, simulate_top
  utils.py           pretty_print_uops

utils/
  __init__.py        public API
  quantization.py    quantize_int8, quantize_int16, dequantize, float_to_bits, bits_to_float

benchmarks/
  harness.py           BenchResult, run_bench
  test_suite.py        correctness test suite (Tier 1/2/3)
  test_perf_suite.py   performance test suite (10 workloads)

tests/
  test_compiler.py   compiler unit tests + multi-kernel integration
  test_top_module.py TopModule hardware simulation tests
  test_add.py / test_relu.py / …   elementwise op tests

docs/
  benchmark_guide.md  ← this file
```
