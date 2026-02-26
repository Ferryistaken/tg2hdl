# tg2hdl

A tinygrad-to-hardware exploration centered on an INT8 GEMV unit in Amaranth.

## Documentation

Docs use **Sphinx + Furo** with automatic API reference generation from Python docstrings (`autodoc`).

### Build locally

```bash
uv sync
uv run sphinx-build -b dirhtml docs docs/_build/dirhtml
```

### Preview locally

```bash
python -m http.server 4173 -d docs/_build/dirhtml
```

Open `http://localhost:4173`.

### Deployment

- GitHub Pages: `.github/workflows/docs-gh-pages.yml`
- Netlify: `netlify.toml`
- Detailed guide: `docs/guide/deployment.md`
