# Deployment

This site now uses **MkDocs Material + mkdocstrings**.

## Local build

```bash
uv sync
uv run mkdocs build
```

Built files are emitted to `site/`.

## Local preview

```bash
uv run mkdocs serve -a 0.0.0.0:4173
```

## GitHub Pages

A workflow is included at `.github/workflows/docs-gh-pages.yml`.

### One-time setup

1. In GitHub, open **Settings → Pages**.
2. Set **Source** to **GitHub Actions**.
3. Push to `main`.

The workflow runs `uv sync`, then `uv run mkdocs build`, then deploys the `site/` folder.

## Netlify

`netlify.toml` is configured to:

- install tooling with `uv sync`
- build with `uv run mkdocs build`
- publish `site`

Just connect this repo in Netlify and deploy.
