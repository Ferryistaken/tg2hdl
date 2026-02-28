# Introduction

## What is tg2hdl?

tg2hdl is a compiler from tinygrad's IR to synthesizable FPGA hardware. You describe a neural network in tinygrad; tg2hdl compiles it to an Amaranth HDL module that simulates cycle-accurately and can be synthesized to an FPGA.

The compiler operates on tinygrad's linearized UOps — the same IR tinygrad uses to emit GPU kernels — and maps each op to hardware: memories, combinational arithmetic, and an FSM sequencer.

## Components

| Path | Role |
|------|------|
| `compiler/backend.py` | `HDLRenderer`, `compile_kernel`, `simulate_kernel` |
| `compiler/hdl_module.py` | `CompiledKernel` — three-pass Amaranth Elaboratable |
| `hdl/gemv.py` | Manual INT8 GEMV unit (reference implementation) |
| `hdl/relu.py` | Combinational ReLU (reference) |
| `tests/test_compiler.py` | Compiler unit and simulation tests |
| `tests/test_combined.py` | Elementwise fusion tests |
| `compare_inference.py` | End-to-end MNIST: CPU float32 vs compiler INT8 |

## Workflow

```
tinygrad model
    │ .schedule()
    ▼
list[ExecItem]
    │ compile_kernel() per SINK item
    ▼
list[CompiledKernel]   (Amaranth Elaboratables)
    │ simulate_kernel()
    ▼
numpy outputs + cycle counts
```

## Status

| Capability | Status |
|------------|--------|
| Generic kernel compilation | ✅ |
| Scalar / elementwise / GEMV patterns | ✅ |
| Fused multi-op kernels (matmul + bias + relu) | ✅ |
| Multi-kernel simulation (Python-chained) | ✅ sim only — see note below |
| Multi-kernel hardware (shared memories, sequenced FSMs) | Planned |
| Multi-MAC parallelism (UNROLL) | Planned |
| FPGA synthesis | Planned |

**Multi-kernel note:** `compile_model()` (`backend.py:143`) compiles each schedule item into an independent `CompiledKernel` with its own private memories. There is no top-level hardware module that wires kernel 0's output memory to kernel 1's input memory or sequences their FSMs. In simulation, `compare_inference.py` bridges the gap by reading kernel 0's output via `buf_read_ports` and writing it into kernel 1's input via `buf_write_ports` — i.e. the Python host acts as the DMA controller. On real hardware, this would require a top-level module that shares BRAMs between kernels and chains their `done`/`start` signals.