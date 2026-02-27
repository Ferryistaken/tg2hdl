# Roadmap

## Current Features (Stable)

### Hardware Primitives
- **GEMV Unit**: Sequential INT8×INT8→INT32 multiply-accumulate
  - Configurable dimensions (M×K)
  - FSM control: IDLE → COMPUTE → EMIT → DONE
  - Cycle-accurate timing: O(M×K)

- **ReLU Unit**: Combinational activation function
  - INT32 input/output
  - Zero-latency (combinational logic)

### Verification Infrastructure
- Bit-accurate simulation via Amaranth simulator
- NumPy reference validation
- Cycle-count assertions
- VCD waveform generation for debugging

### Tooling
- Model training script (`train_mnist.py`)
- Kernel inspection utility (`inspect_kernels.py`)
- Performance benchmarking (`benchmark.py`)

## Work in Progress

### Compiler Backend (`compiler/`)

| Component | Status | Description |
|-----------|--------|-------------|
| `HDLRenderer` | ✅ Complete | Tinygrad renderer for sequential UOps |
| Buffer analysis | ✅ Complete | Extract buffer info from DEFINE_GLOBAL |
| `CompiledKernel` | 🚧 Partial | UOps to Amaranth module conversion |
| `simulate_kernel` | ✅ Complete | Single kernel simulation |
| `compile_model` | 🚧 Partial | Multi-kernel schedule compilation |
| End-to-end model simulation | 🔜 Planned | Full MNIST inference via compiler |

### Target: Auto-Generated GEMV

The compiler backend aims to generate GEMV-like hardware directly from tinygrad's UOp IR, eliminating hand-authored HDL for standard operations.

## Future Roadmap

### Phase 1: Core Hardware Extensions (Q1 2026)

| Feature | Priority | Description |
|---------|----------|-------------|
| Multi-MAC parallelization | High | NUM_MACS parameter for pipelined throughput |
| Bias addition | Medium | Integrated bias vector in GEMV datapath |
| Activation chaining | Medium | GEMV → ReLU → next GEMV without external buffering |
| Full layer execution | Medium | Complete linear+activation layer in hardware |

### Phase 2: Compiler Completion (Q2 2026)

| Feature | Priority | Description |
|---------|----------|-------------|
| UOp pattern matching | High |识别 GEMV, GEMM, convolution patterns |
| Template instantiation | High | Generate parametrized Amaranth from IR |
| Memory optimization | Medium | Shared memory for intermediate activations |
| Kernel fusion | Medium | Combine adjacent operations |

### Phase 3: FPGA Deployment (Q3 2026)

| Feature | Priority | Description |
|---------|----------|-------------|
| Xilinx Vivado export | High | Synthesis constraints, timing closure |
| Lattice ECP5 support | Medium | iCEStudio integration |
| Block RAM inference | Medium | Optimize weight storage |
| DSP utilization | High | Maximize parallel MAC efficiency |

### Phase 4: Model Support Expansion (Q4 2026)

| Feature | Priority | Description |
|---------|----------|-------------|
| Convolution support | High | 2D conv kernels for CNN inference |
| Batch processing | Medium | Support batch > 1 without redesign |
| Dynamic shapes | Low | Runtime-configurable dimensions |
| FP16/INT4 support | Low | Alternative quantization schemes |

## Long-Term Vision

### Research Directions

1. **Automatic Parallelization**: Static analysis to determine optimal NUM_MACS for target FPGA
2. **Memory Hierarchy**: Multi-level caching for large models (weights vs activations)
3. **Power Gating**: Fine-grained power management for edge deployment
4. **Model-Specific Optimization**: Tailored architectures for transformer blocks, RNNs

### Integration Targets

- **Edge TPU Alternative**: Open-source FPGA deployment for small models
- **Tinygrad Hardware Backend**: First-class HDL support in tinygrad ecosystem
- **Educational Platform**: Demonstrable ML hardware design flow

## Contribution Areas

| Area | Difficulty | Description |
|------|------------|-------------|
| Multi-MAC GEMV | Medium | Extend single-MAC to parallel design |
| Compiler UOp patterns | Hard | Map UOps to hardware templates |
| FPGA constraints | Medium | Timing constraints for real deployment |
| Additional activations | Easy | SiLU, GELU, etc. |
| Benchmark expansion | Easy | More models, more platforms |