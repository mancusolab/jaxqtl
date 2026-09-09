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

Use ordinary prose by default. Add a notice when its consequence or action deserves attention:

| Type | Use for | Title and body |
| --- | --- | --- |
| `note` | Non-obvious behavior needed to interpret an input or result | Factual title; behavior → implication |
| `tip` | An optional shortcut or improvement | Action title; action → benefit → tradeoff |
| `warning` | A choice that risks incorrect inference or an avoidable operational failure | Preventive title; condition → consequence → action |
| `failure` | An actual failed operation in troubleshooting | Symptom title; cause or diagnosis → recovery |
| `danger` | Destructive actions that can lose data or work | Preventive title; loss at risk → safe alternative |

Use these names, not icon-based aliases such as `fire`, `error`, or `caution`. Reserve `abstract` for collapsible
API contracts. A strong recommendation needed for inference quality, such as SPA with score-test ACAT, is a
`warning`, not an optional `tip`.

Keep titles in sentence case and bodies to one issue in two or three short sentences. Put notices beside the
relevant action; never collapse consequential warnings. Repeat essential warnings on independently usable
workflows, but link to the methods page for their full explanation. Routine setup instructions do not need a box.

## Writing for limited attention

Give each page one job: getting started runs an example, workflows guide decisions, reference pages define exact
contracts, and troubleshooting explains symptoms. Keep complete commands where users run them. Explain a caveat
once per page, link to deeper detail, and avoid repeating the same checklist in prose, a table, and a notice.
