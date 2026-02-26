# Architecture

## Dataflow

The present network path follows:

1. `h = (x @ w1 + b1).relu()`
2. `logits = h @ w2 + b2`

Because input is batch size 1, these are matrix-vector products. The hardware primitive is therefore GEMV.

## GEMV unit (`hdl/gemv.py`)

The unit stores:

- Vector memory `x[K]` (INT8)
- Weight memory `W[M×K]` (INT8, row-major)
- Accumulator `acc` (INT32)

### FSM

- `IDLE`: waits for `start`.
- `COMPUTE`: one MAC per cycle for `K` columns.
- `EMIT`: publishes `result_data` for the active row.
- `DONE`: raises `done` and returns to idle.

### Interface highlights

- Write ports for loading vector and weight memories.
- Streaming-like result channel via `result_valid`, `result_idx`, `result_data`.
- Control signals: `start`, `busy`, `done`.

## Timing model

For a single-MAC design:

- Per row: `K` compute cycles + 1 emit cycle
- Total: `M × (K + 1)`

At 100 MHz, kernel timing is straightforward to estimate from cycle count.
