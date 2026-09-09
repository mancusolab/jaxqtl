# Contributing

See [Community guidelines](conduct.md) for participation and reporting expectations.

## Development setup

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
uv sync --frozen --extra dev --extra docs
uv run zensical build --strict
```

Generated HTML is written to `site/` and is not tracked. Documentation source, including every Python API page under
`docs/api/`, must be committed so a clean CI checkout can reproduce the site.

Theme overrides and Python API templates live in `docs_theme/`, outside the published `docs/` content tree.

Preview the site locally with `uv run zensical serve`. Check the user-guide navigation, API signatures, internal
links, and equations when changing documentation. Describe the supported API directly, and verify examples
against the source code and CLI defaults.

!!! danger "Keep analysis artifacts out of the documentation cache"

    Zensical's `--clean` option deletes the repository's `.cache/` directory, including any analysis artifacts
    stored there. Use the command above for local builds. Run a clean build only in a disposable checkout, and
    store benchmark logs and scientific results outside build caches. Deployment uses a fresh CI checkout.

## API page headings

Use one descriptive page title and sentence-case subsections that name the topic, such as "Genotype loading" or
"Covariance estimators". Omit generic "API" and "Implementations" headings. Short pages with one coherent set of
symbols may list them directly after the introduction; add subsections only to distinguish topics. Let generated
symbol headings identify individual classes and functions rather than repeating their names in manual headings.

## Documentation notices

Choose admonitions by meaning, not color or icon. Use these canonical types consistently:

| Type | Purpose | Internal structure |
| --- | --- | --- |
| `note` | Non-obvious behavior or scope the reader needs to understand | Factual title; behavior, then its implication |
| `tip` | Recommended action or useful shortcut | Action title; recommendation, then benefit and relevant limitation |
| `warning` | A choice that can invalidate an analysis or cause an avoidable operational problem | Action title; risky condition, consequence, then prevention |
| `failure` | An actual error or failed operation in troubleshooting | Symptom title; what failed, then how to diagnose or recover |
| `danger` | Destructive actions or risk of losing data or work | Preventive action title; destructive consequence, then a safe alternative |

Use the canonical type names above rather than selecting blocks by their icons or alternating aliases such as
`error`, `bug`, or `caution` for visual variety. Performance advice belongs in a `tip` only when it recommends an
action and explains the tradeoff. `abstract` is reserved for collapsible API contracts, not user notices.

Use sentence-case titles and short prose bodies. A notice should address one issue. Prefer ordinary prose for
routine options and background details; do not wrap every caveat in a box. Keep consequential warnings visible
beside the relevant action, never inside collapsed details. Keep repeated notices consistent across independently
usable workflows, and link to the detailed explanation rather than repeating it in full.
