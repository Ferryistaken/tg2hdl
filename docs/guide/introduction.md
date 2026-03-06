# Introduction

## What is tg2hdl?

tg2hdl is a compiler from tinygrad's IR to synthesizable FPGA hardware. You describe a neural network in tinygrad; tg2hdl compiles it to an Amaranth HDL module that simulates cycle-accurately and can be synthesized to an FPGA.

The compiler operates on tinygrad's linearized UOps — the same IR tinygrad uses to emit GPU kernels — and maps each op to hardware: memories, combinational arithmetic, and an FSM sequencer.

## Components

| Path | Role |
|------|------|
| `compiler/backend.py` | `HDLRenderer`, `compile_kernel`, `compile_model`, `compile_top_module`, `simulate_kernel`, `count_cycles_from_schedule` |
| `compiler/hdl_module.py` | `CompiledKernel` — three-pass Amaranth Elaboratable (KernelIR → hardware) |
| `compiler/ir.py` | Typed IR: `DType`, `IRConst/Counter/BufLoad/RegLoad/Op`, `IRBufStore/RegStore`, `LoopIR`, `BufferMeta`, `KernelIR` |
| `compiler/uop_to_ir.py` | `uop_to_ir()` — single-pass UOp list → `KernelIR` conversion |
| `compiler/lowering/arithmetic.py` | `ArithmeticLowering`, `create_counters()` — combinational signal emission |
| `compiler/lowering/control.py` | `build_control()` — FSM construction from typed loop tree |
| `compiler/fp32.py` | `FP32Add`, `FP32Mul`, `FP32Cmp` — IEEE 754 combinational hardware modules |
| `compiler/top_module.py` | `TopModule`, `simulate_top` — multi-kernel sequencer with copy FSM |
| `compiler/utils.py` | `pretty_print_uops` — UOp inspection helper |
| `benchmarks/harness.py` | `run_bench`, `BenchResult` — compare any tinygrad graph vs HDL simulation |
| `benchmarks/test_suite.py` | Correctness suite: Tier 1–3 (elementwise, GEMV, multi-kernel MLP) |
| `benchmarks/test_perf_suite.py` | Performance suite: 10 workloads, scalar → MNIST-scale |
| `utils/quantization.py` | `quantize_int8`, `dequantize` — user-level quantization helpers |
| `tests/test_compiler.py` | Compiler unit and simulation tests |
| `tests/test_top_module.py` | TopModule hardware simulation tests |
| `tests/test_fp32.py` | IEEE 754 FP32 unit and integration tests |
| `compare_inference.py` | End-to-end MNIST: CPU float32 vs compiler INT8 |

## Workflow

```
tinygrad model
    │ .schedule()
    ▼
list[ExecItem]
    │ compile_top_module()  ← auto-detects inter-kernel connections
    ▼
TopModule + list[KernelSpec]     (Amaranth Elaboratables)
    │ simulate_kernel() per kernel   — or —   simulate_top()
    ▼
numpy outputs + cycle counts
```

Or via the benchmark harness (handles single- and multi-kernel automatically):

```python
from benchmarks.harness import run_bench
result = run_bench("my_kernel", build_fn, input_arrays)
assert result.correct
```

## Status

| Capability | Status |
|------------|--------|
| Generic kernel compilation | ✅ |
| Scalar / elementwise / GEMV patterns | ✅ |
| Fused multi-op kernels (matmul + bias + relu) | ✅ |
| Multi-kernel hardware sequencing (`TopModule`) | ✅ |
| Float32 — IEEE 754 hardware simulation | ✅ `FP32Add`, `FP32Mul`, `FP32Cmp` |
| Float16 / BFloat16 arithmetic | ❌ No dedicated units — compile error in practice |
| Multi-MAC parallelism (UNROLL) | Planned |
| FPGA synthesis | Planned |

## Supported ops

The compiler handles: `ADD`, `MUL`, `CAST`, `CMPLT`, `WHERE`, `MAX`, `LOAD`, `STORE`, `RANGE`, `INDEX`, `DEFINE_GLOBAL`, `DEFINE_REG`, `CONST`, `AFTER`.

All other UOps raise `NotImplementedError` at compile time (fail-loud policy).
