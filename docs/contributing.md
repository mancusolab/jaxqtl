# Development setup

Clone the repository and synchronize the development environment:

```bash
git clone https://github.com/mancusolab/jaxqtl.git
cd jaxqtl
uv sync --frozen --extra dev
```

Run the test suite with the repository-required capture setting:

```bash
uv run pytest -p no:capture
```

## Build the documentation

Install the documentation dependencies and run a strict build:

```bash
uv sync --frozen --extra docs
uv run zensical build --strict --clean
```

Generated HTML is written to `site/` and is not tracked. Documentation source, including every Python API page under
`docs/api/`, must be committed so a clean CI checkout can reproduce the site.
