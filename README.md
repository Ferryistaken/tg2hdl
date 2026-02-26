# tg2hdl

A tinygrad-to-hardware exploration centered on an INT8 GEMV unit in Amaranth.

## Documentation

Docs use **MkDocs Material** with automatic API reference generation from Python docstrings (`mkdocstrings`).

### Run locally

```bash
uv sync
uv run mkdocs serve -a 0.0.0.0:4173
```

Open `http://localhost:4173`.

### Build static docs

```bash
uv run mkdocs build
```

Output is in `site/`.

### Deployment

- GitHub Pages: `.github/workflows/docs-gh-pages.yml`
- Netlify: `netlify.toml`
- Detailed guide: `docs/guide/deployment.md`
