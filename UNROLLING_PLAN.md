# Plan: Phase 2 — LOOP-Axis Unrolling & PE Replication (Priority #1)

## Context

### Problem
Today the compiler executes both `LOOP` and `REDUCE` ranges sequentially in the FSM. This means even independent output-element work is serialized, so GEMV/GEMM throughput is capped by a single effective MAC lane and software-style loop trip counts. The architecture docs already preserve `AxisType` metadata (LOOP vs REDUCE), but the backend currently does not exploit LOOP-level parallelism.

### Goal
Introduce safe, incremental hardware parallelism by unrolling `LOOP` axes and replicating processing elements (PEs), while preserving existing correctness and deterministic cycle accounting for `REDUCE` accumulation semantics.

### Non-goals (for this phase)
- No REDUCE-axis parallel tree reduction yet.
- No major rewrite of tinygrad frontend integration.
- No broad auto-scheduler initially; start with static/manual unroll factors.

---

## North-star outcome

For LOOP-dominant kernels (elementwise, output-channel loops in GEMV/GEMM), cycle counts should scale approximately:

- baseline: `cycles ~= M * (K + overhead)`
- unrolled by `U` over LOOP axis: `cycles ~= ceil(M/U) * (K + overhead')`

with bounded overhead for tail handling and memory/bus arbitration.

---

## Deliverables

1. **IR + lowering support for lane-aware LOOP unrolling**
2. **Datapath replication across U lanes (PEs)**
3. **Control FSM updates for batched LOOP index stepping + tail masking**
4. **Memory/read/write port strategy for multi-lane loads/stores**
5. **Verification + perf test expansion**
6. **CLI/benchmark knobs for unroll factor sweep**

---

## Step-by-step implementation plan

## Step 0 — Baseline, guardrails, and observability

### 0.1 Add explicit compiler options surface
- Add `unroll_loop: int = 1` (default preserves existing behavior).
- Thread option from high-level compile entry points down to:
  - UOp→IR conversion context (or post-IR transform)
  - arithmetic lowering
  - control lowering

### 0.2 Add capability checks + early fallback
- Reject unroll (`>1`) for unsupported patterns initially:
  - non-unit LOOP stride assumptions not yet modeled
  - conflicting memory access patterns that cannot be lane-separated
- Fallback to `unroll=1` with a clear debug message (not silent miscompile).

### 0.3 Add perf observability fields
- Emit lane-aware metadata in compile reports:
  - unroll factor
  - estimated LOOP iterations per cycle
  - tail length (`bound % U`)

---

## Step 1 — IR transform for LOOP-axis unrolling

### 1.1 Introduce a dedicated IR transform pass
Add `LoopUnrollTransform` between `uop_to_ir` and arithmetic lowering.

Inputs:
- `KernelIR` with loop tree + typed stores/values
- target unroll factor `U`

Outputs:
- transformed `KernelIR` where one LOOP body represents U logical lanes
- lane-indexed expressions for address/value generation

### 1.2 Unroll strategy
For a target LOOP node with bound `N`:
- new loop bound: `N_main = N // U`
- generate lane clones for `lane in [0..U-1]`
- each lane substitutes original loop index `i` with `i_base + lane`
- optionally add tail path for `N_tail = N % U`

### 1.3 Tail policy (initial)
- Keep a scalar/tiny-loop epilogue for `N_tail` (simple + correct).
- Later optimization: predicate lanes with valid mask.

### 1.4 Safety constraints
- Only unroll `AxisType.LOOP` nodes in first iteration.
- Do not alter REDUCE loop structure.

---

## Step 2 — Arithmetic lowering with lane vectors

### 2.1 Lane signal model
Extend lowering result structure to support per-lane signals:
- from `value_id -> Signal`
- to `value_id -> list[Signal]` (size U) for unrolled regions

### 2.2 Clone datapath per lane
For each unrolled lane:
- instantiate arithmetic subgraph independently
- map loads/stores to lane-local addresses
- keep dtype handling unchanged

### 2.3 FP32/INT datapath handling
- INT path: straightforward replication.
- FP32 path: replicate FP32 modules per lane (`FP32Add`, `FP32Mul`, `FP32Cmp`) with resource impact tracked.

### 2.4 Register semantics
If lane writes target independent output elements:
- either lane-local accumulators or combinational lane outputs depending on original IR role.

For REDUCE regions nested under unrolled LOOP:
- each lane keeps independent accumulator state (same reduce trip count, separate data).

---

## Step 3 — Control FSM updates

### 3.1 Counter progression
For unrolled LOOP axis:
- loop counter increments by 1 over `N_main`, but represents a base index for U lanes.

### 3.2 Store emission
- emit up to U stores per iteration for LOOP outputs.
- route lane writes with deterministic ordering/enable.

### 3.3 Tail execution
After main unrolled section:
- run tail states for `N_tail` elements through existing scalar path.

### 3.4 Done semantics
- preserve one-cycle `done` pulse behavior.
- ensure done waits for tail completion.

---

## Step 4 — Memory architecture for multi-lane access

### 4.1 First implementation (functional)
Use replicated reads/writes with conservative arbitration:
- if port limits prevent true parallel accesses, serialize at port boundary (correctness first).

### 4.2 Practical optimization path
- bank memories by low address bits for natural lane partitioning.
- generate one read port per bank where feasible.
- ensure conflict handling is deterministic (stall or lane-serialize).

### 4.3 Write conflict contract
For unrolled LOOP only, enforce no two lanes write same address; if detected, fall back to unroll=1 (initially).

---

## Step 5 — Verification strategy

### 5.1 Unit tests (new)
- IR transform tests:
  - bound divisible by U
  - bound non-divisible (tail)
  - nested LOOP+REDUCE
- Lowering tests:
  - per-lane signal existence and dtype consistency
- Control tests:
  - cycle counts match expected formulas with/without tail

### 5.2 End-to-end correctness tests
For each supported op family, sweep `U in {1,2,4,8}`:
- elementwise add/relu
- GEMV int8→int32
- FP32 elementwise + GEMV

Assertions:
- outputs equal baseline (`U=1`)
- integer exactness preserved
- float tolerance unchanged

### 5.3 Regression tests
- compile fallback cases trigger clear warning and preserve correctness.
- ensure existing tests remain green with default `U=1`.

---

## Step 6 — Benchmarking and rollout

### 6.1 Harness integration
- add unroll parameter to benchmark harness and perf suite.
- report:
  - cycles
  - cycles-per-output
  - speedup vs `U=1`

### 6.2 Compare script integration
- optional `--unroll-loop` arg for FPGA path only.
- keep CPU/GPU baselines unchanged.

### 6.3 Milestones
- **M1**: elementwise LOOP unroll works, tested, measurable speedup.
- **M2**: GEMV output-axis LOOP unroll with REDUCE untouched.
- **M3**: basic memory banking to recover near-linear scaling for int8 GEMV.

---

## Risks and mitigations

### Risk 1: Memory-port bottlenecks erase speedup
- Mitigation: explicit banking + conflict stats; document achieved effective lane utilization.

### Risk 2: Control complexity and tail bugs
- Mitigation: strict transform-level tests + cycle golden tests for small bounds.

### Risk 3: Area explosion (especially FP32)
- Mitigation: cap default `U`, expose tunable, report per-U resource estimates.

### Risk 4: Unsupported address patterns
- Mitigation: conservative pattern checks + fallback to `U=1`.

---

## Suggested first implementation order (engineering sequence)

1. Add compile option + pass plumbing (`U=1` no-op).
2. Implement LOOP unroll transform for single-level LOOP kernels only.
3. Replicate INT elementwise datapath and stores.
4. Add tail handling.
5. Extend to GEMV output LOOP + per-lane REDUCE accumulators.
6. Add FP32 replication.
7. Add banking and conflict logic.
8. Benchmark sweep + docs.

---

## Success criteria

- Correctness parity with baseline for all existing tests at `U=1` and new unroll tests at `U>1`.
- Demonstrated cycle reduction on LOOP-dominant kernels:
  - >=1.7× at `U=2`
  - >=3.0× at `U=4`
  (allowing overhead/port limits)
- Clear compiler diagnostics for fallback cases.
- No regressions in default flow.
