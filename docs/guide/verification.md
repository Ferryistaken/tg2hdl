# Verification

Amaranth cycle-accurate simulation against NumPy ground truth. All numeric results are compared at the bit level — the numpy references model INT8 truncation at every intermediate step to match UOp semantics exactly.

## Test suite

### Compiler — structure (`tests/test_compiler.py`)

| Test | Validates |
|------|-----------|
| `test_renderer_produces_clean_uops` | No GPU-specific ops in UOp output |
| `test_renderer_attributes` | `has_local=False`, `has_shared=False`, etc. |
| `test_matmul_buffers` | Buffer depth/width/index analysis |
| `test_scalar_returns_root_no_body` | Parser handles no-RANGE kernels |
| `test_elementwise_one_loop` | Parser: single LOOP level |
| `test_gemv_two_levels` | Parser: LOOP + REDUCE nesting |
| `test_gemv_prologue_epilogue_split` | acc reset in prologue, output write in epilogue |
| `test_mnist_kernel_shapes` *(slow)* | MNIST schedule produces 2 kernels with correct buffer shapes |

### Compiler — simulation (`tests/test_compiler.py`, `tests/test_combined.py`)

| Test | Validates |
|------|-----------|
| `test_compile_small_matmul` | Kernel object has expected ports |
| `test_simulate_identity_matmul` | 3×3 identity: output == input |
| `test_simulate_small_matmul_4x3` | 4×3 matmul matches numpy with INT8 truncation |
| `test_simulate_matmul_with_bias_relu` | Fused matmul + bias + ReLU |
| `test_cycle_count` | M×(K+2) cycle model |
| `test_relu_add_bias_*` (4 tests) | Elementwise fusion: relu(a+b+const), N-cycle throughput |
| `test_two_layer_mlp_simulation` | Two kernels chained, both outputs match numpy |
| `test_two_layer_mlp_prediction` | Larger random MLP, argmax matches |

## Running tests

```bash
uv run pytest                                  # full suite
uv run pytest -k "not slow"                    # skip MNIST shape test
uv run pytest tests/test_compiler.py -v        # compiler only
uv run pytest tests/test_compiler.py tests/test_combined.py -v   # compiler + elementwise
```