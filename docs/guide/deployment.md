# Deployment

This site now uses **Sphinx + Furo theme + autodoc**.

## Local build

```bash
uv sync
uv run sphinx-build -b dirhtml docs docs/_build/dirhtml
```

Built files are emitted to `docs/_build/dirhtml`.

## Local preview

```bash
python -m http.server 4173 -d docs/_build/dirhtml
```

## GitHub Pages

A workflow is included at `.github/workflows/docs-gh-pages.yml`.

### One-time setup

1. In GitHub, open **Settings → Pages**.
2. Set **Source** to **GitHub Actions**.
3. Push to `main`.

The workflow runs `uv sync`, then `uv run sphinx-build -b dirhtml docs docs/_build/dirhtml`, then deploys `docs/_build/dirhtml`.

## Netlify

`netlify.toml` is configured to:

- install tooling with `python -m uv sync --frozen`
- build with `python -m uv run sphinx-build -b dirhtml docs docs/_build/dirhtml`
- publish `docs/_build/dirhtml`

Just connect this repo in Netlify and deploy.


## Troubleshooting local build warnings

If you previously used npm/VitePress in this repo, remove stale `docs/node_modules` and `docs/.vitepress` directories.
These can confuse docs tooling during migration.

```bash
python - <<'P'
import shutil, pathlib
for p in ['docs/node_modules', 'docs/.vitepress']:
    q = pathlib.Path(p)
    if q.exists():
        shutil.rmtree(q)
print('cleaned stale frontend artifacts')
P
```
