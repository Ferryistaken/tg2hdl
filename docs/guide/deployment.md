# Deployment

This page shows how to build and host the docs on **GitHub Pages** and **Netlify**.

## Local production build

```bash
cd docs
npm ci
npm run docs:build
```

Generated output is in `docs/.vitepress/dist`.

## GitHub Pages

### Option A: Automatic deploy with GitHub Actions (recommended)

This repo includes a workflow at `.github/workflows/docs-gh-pages.yml` that:

1. installs docs dependencies,
2. builds VitePress,
3. deploys to GitHub Pages.

### One-time setup

In your GitHub repo:

1. Go to **Settings → Pages**.
2. Under **Build and deployment**, choose **Source: GitHub Actions**.
3. Push to `main` (or your default branch).

The workflow sets `DOCS_BASE=/<repo-name>/` automatically so routes work on Pages project sites.

## Netlify

A `netlify.toml` is included at repo root with:

- base directory: `docs`
- build command: `npm ci && npm run docs:build`
- publish directory: `.vitepress/dist`

### Deploy steps

1. Create a new Netlify site and connect this GitHub repo.
2. Netlify should auto-detect `netlify.toml`.
3. Trigger deploy.

No custom base path is required for Netlify in the default setup.

## Notes on base paths

- VitePress `base` is configured via `DOCS_BASE` env var in `docs/.vitepress/config.mjs`.
- Default is `/` for local dev and Netlify.
- GitHub Pages build sets it to `/<repo-name>/`.
