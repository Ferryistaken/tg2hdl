# Verification

## Simulation strategy

The project uses Amaranth simulation as a bit-accurate testbench environment:

- preload vector/weights,
- pulse `start`,
- collect `result_valid` outputs,
- compare with NumPy reference GEMV.

## Coverage highlights

`tests/test_gemv.py` includes:

- small deterministic cases (`2x2`, `4x3`),
- signed arithmetic checks with negatives,
- identity matrix behavior,
- single-element edge case,
- randomized INT8 validation,
- cycle-count assertion,
- MNIST-sized kernel (`10x128`, marked slow),
- ReLU combinational behavior checks.

## Suggested command

```bash
uv run pytest
```

For quick iteration, run a subset:

```bash
uv run pytest tests/test_gemv.py -k "not slow"
```
