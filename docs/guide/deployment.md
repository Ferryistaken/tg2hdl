# Deployment

## Documentation Build System

The documentation uses Sphinx with Furo theme and autodoc for API reference generation.

### Local Build

```bash
uv sync
uv run sphinx-build -b dirhtml docs docs/_build/dirhtml
```

Output: `docs/_build/dirhtml/`

### Local Preview

```bash
python -m http.server 4173 -d docs/_build/dirhtml
```

Access: `http://localhost:4173`

## GitHub Pages

Workflow: `.github/workflows/docs-gh-pages.yml`

### Configuration

1. Repository Settings → Pages
2. Source: **GitHub Actions**
3. Push to main branch

The workflow automatically builds and deploys on each push.

## Netlify

Configuration: `netlify.toml`

### Setup

1. Connect repository in Netlify dashboard
2. Deploy (build command and publish directory auto-detected)

### Build Configuration

```toml
[build]
  command = "python -m uv run sphinx-build -b dirhtml docs docs/_build/dirhtml"
  publish = "docs/_build/dirhtml"
```