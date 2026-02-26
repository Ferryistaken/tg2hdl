# tg2hdl

A tinygrad-to-hardware exploration focused on an INT8 GEMV accelerator in Amaranth.

!!! info "What you get in these docs"
    - High-level project guides (motivation, architecture, verification, deployment)
    - Auto-generated API reference from Python docstrings (`mkdocstrings`)
    - Reproducible local/CI build flow with `uv` + `mkdocs`

## Start here

- [Introduction](guide/introduction.md)
- [Getting Started](guide/getting-started.md)
- [Architecture](guide/architecture.md)

## Build docs locally

```bash
uv sync
uv run mkdocs serve -a 0.0.0.0:4173
```

Then open `http://localhost:4173`.

## API docs

- [API Reference (auto-generated)](guide/api-reference.md)
