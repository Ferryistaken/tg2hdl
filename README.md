# tg2hdl

A tinygrad-to-hardware exploration centered on an INT8 GEMV unit in Amaranth.

## Documentation

A modern VitePress docs site is included under [`docs/`](docs), with an extensive project introduction first:

- Introduction: `docs/guide/introduction.md`
- Getting Started: `docs/guide/getting-started.md`
- Architecture: `docs/guide/architecture.md`
- Verification: `docs/guide/verification.md`

```bash
cd docs
npm install
npm run docs:dev
```

Open `http://localhost:4173`.


## Deployment

- GitHub Pages: workflow at `.github/workflows/docs-gh-pages.yml` (set Pages source to GitHub Actions).
- Netlify: config at `netlify.toml` (auto-detected in most setups).
- Full instructions: `docs/guide/deployment.md`.
