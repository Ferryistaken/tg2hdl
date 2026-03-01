# Verification

Amaranth cycle-accurate simulation against NumPy ground truth. Integer results are compared at the bit level — the numpy references model INT8 truncation at every intermediate step to match UOp semantics exactly. Float32 results are compared with `rtol=1e-5` against tinygrad CPU reference.

## Test suite

```bash
uv run pytest tests/ benchmarks/ -k "not slow" -v   # 93 tests, ~3 s
uv run pytest tests/ benchmarks/ -v                  # all tests incl. slow
```

### FP32 unit tests (`tests/test_fp32.py`) — 38 tests

| Class | What it validates |
|-------|-------------------|
| `TestFP32Add` | Positive/negative/mixed add, cancellation, zero, large exponent diff, infinity, 5 parametrized cases |
| `TestFP32Mul` | All sign combinations, mul-by-zero, mul-by-one, infinity, 6 parametrized cases |
| `TestFP32Cmp` | Less-than for positive, negative, mixed sign, ±0, 5 parametrized cases |
| Integration | `test_fp32_relu_harness`, `test_fp32_add_harness` — float32 through full `run_bench` |

### Compiler — structure (`tests/test_compiler.py`)

| Test | Validates |
|------|-----------|
| `test_renderer_produces_clean_uops` | No GPU-specific ops in UOp output |
| `test_renderer_attributes` | `has_local=False`, `has_shared=False`, etc. |
| `test_matmul_buffers` | Buffer depth/width/index analysis |
| `test_scalar_returns_root_no_body` | `uop_to_ir` handles no-RANGE kernels |
| `test_elementwise_one_loop` | IR: single LOOP level, prologue stores present |
| `test_gemv_two_levels` | IR: LOOP + REDUCE nesting, correct bounds |
| `test_gemv_prologue_epilogue_split` | `IRRegStore` in prologue, `IRBufStore` in epilogue |
| `test_mnist_kernel_shapes` *(slow)* | MNIST schedule produces 2 kernels with correct buffer shapes |

### Compiler — simulation (`tests/test_compiler.py`)

| Test | Validates |
|------|-----------|
| `test_compile_small_matmul` | Kernel object has expected ports |
| `test_simulate_identity_matmul` | 3×3 identity: output == input |
| `test_simulate_small_matmul_4x3` | 4×3 matmul matches numpy with INT8 truncation |
| `test_simulate_matmul_with_bias_relu` | Fused matmul + bias + ReLU |
| `test_cycle_count` | M×(K+2) cycle model |
| `test_two_layer_mlp_simulation` | Two kernels chained, both outputs match numpy |
| `test_two_layer_mlp_prediction` | Larger random MLP, argmax matches |

### Elementwise fusion (`tests/test_relu.py`, `tests/test_combined.py`)

| Test | Validates |
|------|-----------|
| `test_relu_*` | relu over all-positive, all-negative, mixed int32 |
| `test_relu_add_bias_*` (4 tests) | `relu(a+b+const)`, N-cycle throughput |

### TopModule (`tests/test_top_module.py`) — 9 tests

| Test | Validates |
|------|-----------|
| `test_connections_detected` | Auto-detection of buffer identity connections |
| `test_ext_write_ports_exposed` | Non-connected inputs exposed correctly |
| `test_output_rport_wired` | Final kernel output accessible |
| `test_simulate_top_*` | End-to-end TopModule simulation with 2-layer MLP |

### Benchmark suites

```bash
# Correctness suite — 8 tests: Tier 1 elementwise, Tier 2 GEMV, Tier 3 MLP
uv run pytest benchmarks/test_suite.py -v

# Performance suite — 10 workloads with cycle-count validation
uv run pytest benchmarks/test_perf_suite.py -v -s -k "not slow"
```

Performance suite workloads:

| Test | Shape | Expected cycles |
|------|-------|----------------|
| scalar add | (1,) | ≤ 5 |
| elementwise relu | N=32 | ~33 |
| elementwise add+relu | N=128 | ~129 |
| tiny GEMV int8 | (1,4)@(4,8) | 49 |
| small GEMV+bias int8 | (1,8)@(8,16) | 161 |
| linear+bias+relu int8 | (1,8)@(8,16) | 161 |
| 2-layer MLP small | (1,4)→(1,4)→(1,2) | ~36 |
| 2-layer MLP medium | (1,16)→(1,16)→(1,8) | ~432 |
| relu fp32 | N=16 | 16 |
| add fp32 | N=32 | 32 |
| gemv fp32 | (1,4)@(4,8) | 48 |
| MNIST layer 1 *(slow)* | (1,784)@(784,128) | 100,609 |
