# TODO — Hardware/Software Co-Design Directions

## Known Issues

### Compiler conflates quantization with compilation

`quantize_int8()` lives in `compiler/backend.py` and is exported from
`compiler/__init__.py`, implying quantization is the compiler's job.  It
isn't.  The compiler is already dtype-agnostic (`_dtype_to_width()` handles
int8/int16/int32 etc.).  Quantization is a user-level choice expressed in the
tinygrad graph (`dtypes.int8`).

**Suggested fix:**
- Move `quantize_int8()` out of the compiler public API into a separate
  `utils/quantization.py` helper.
- Remove from `compiler/__init__` exports.
- Add float16/bfloat16 signal-width support to `_dtype_to_width()`.
- Tests and benchmarks should be dtype-agnostic, not assume INT8.

---

## Future Directions

### UNROLL → multi-MAC datapath
tinygrad can emit `UNROLL` UOps (currently suppressed by `NOOPT=1`).
Consuming these would let the compiler instantiate N parallel MACs instead of
one sequential MAC per cycle.  This requires:
- Removing `NOOPT=1` or selectively enabling UNROLL.
- Extending `_build_datapath` to handle `Ops.EXPAND` / `Ops.UNROLL`.
- Parameterising `CompiledKernel` with a MAC-parallelism factor.

**Impact:** K× throughput improvement for the inner reduction loop.

### Shared BRAMs (zero-copy kernel chaining)
The current `TopModule` copies kernel A's output Memory into kernel B's input
Memory one element per clock cycle.  For large buffers this latency dominates.

Kernel A and kernel B could share the **same** Amaranth `Memory` object:
- `CompiledKernel` would accept an optional externally-provided `Memory` for
  each buffer instead of always allocating its own.
- `TopModule` would pass kernel A's output memory directly as kernel B's input
  memory, eliminating the copy FSM state.
- Savings: `depth` copy cycles per kernel boundary.

### Hardware-aware scheduling
tinygrad's scheduler makes fusion/split decisions based on op counts and memory
traffic, but knows nothing about FPGA resources (BRAM count, DSP budget,
routing congestion).

Options:
- Pass an FPGA resource budget to `compile_model` and use it to limit kernel
  fusion (avoid creating kernels that exceed BRAM capacity).
- Expose a `tile_size` parameter that maps to the inner loop bound, controlling
  how many DSPs are used per cycle.
- Integrate with vendor tools (Vivado, Quartus) to close the loop on actual
  resource utilisation.

### Quantization-aware training
The current flow quantizes a pre-trained float32 model post-hoc.  A
quantization-aware training (QAT) approach would:
- Express hardware truncation semantics as differentiable tinygrad ops
  (straight-through estimator for `cast(int8)`).
- Train directly in the int8 domain, closing the accuracy gap between the
  float reference and the HDL simulation.
- Potentially allow the compiler to guarantee monotone accuracy: simulation
  error ≤ hardware error ≤ float reference error.

### CI/CD regression tracking
The benchmark suite (`benchmarks/suite.py`) currently asserts correctness but
does not track performance regressions.  Future additions:
- Record `hdl_cycles` and `sim_wall_s` in a JSON artefact per CI run.
- Plot cycle count vs commit graph (e.g. via GitHub Actions artefact + a small
  visualisation step).
- Alert on >5 % cycle-count regression relative to the rolling baseline.
