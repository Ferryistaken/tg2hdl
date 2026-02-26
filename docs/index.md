# tg2hdl

Tinygrad-to-hardware exploration focused on an INT8 GEMV accelerator in Amaranth.

## Documentation map

```{toctree}
:maxdepth: 2
:caption: Guide

guide/introduction
guide/getting-started
guide/architecture
guide/verification
guide/deployment
guide/api-reference
```

## Quickstart

```bash
uv sync
uv run sphinx-build -b dirhtml docs docs/_build/dirhtml
```

Then open `docs/_build/dirhtml/index.html`.
