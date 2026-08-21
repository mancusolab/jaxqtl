# Test requirements: negative-binomial initializer offset

**Status:** Pre-execution requirements

**Scope:** Item 2 only

## Traceability matrix

| Test ID | Acceptance criteria | Verification type | Expected file |
|---|---|---|---|
| NB-OFFSET-001 | AC1.1, AC1.2 | Parameterized numerical regression | `tests/test_lm_glm.py` |
| NB-OFFSET-002 | AC2.1 | Strict documentation build and semantic inspection | `src/jaxqtl/infer/_glm.py`, `src/jaxqtl/infer/_optimize.py` |
| NB-OFFSET-003 | AC3.1 | Existing regression suites and scoped diff inspection | Item-2 source and test files |

## NB-OFFSET-001: Initializer offset and dispersion contract

Add `tests/test_lm_glm.py::test_negative_binomial_initializer_returns_predictor_without_offset`, parameterized over a
nonzero scalar offset and a nonconstant `(n,)` vector offset. Use deterministic finite data, the default
negative-binomial log link, and identical solver and optimizer settings for `_NBInit` and the independently fitted
preliminary Poisson model.

The test must:

1. Call `_NBInit.init(X, y, offset, ...)`.
2. Independently fit the equivalent `GeneralizedLinearModel(family=Poisson(), ...)`.
3. Treat `poisson_state.eta` as the complete preliminary predictor.
4. Independently reproduce the existing moment estimate from that complete predictor.
5. Call `NegativeBinomial.estimate_dispersion` with the complete predictor, the reciprocal moment estimate, and the
   same `max_iter`, then apply the existing `jnp.nan_to_num(..., nan=0.1)` fallback.
6. Assert the initializer dispersion matches that independent result before testing the predictor.
7. Assert `initializer_eta + offset == poisson_state.eta`.

Run:

```bash
uv run pytest -p no:capture tests/test_lm_glm.py::test_negative_binomial_initializer_returns_predictor_without_offset -q
```

RED requires both dispersion assertions to pass followed by two predictor failures whose differences equal the supplied
offsets. GREEN requires both parameterized cases to pass. Do not assert iteration counts, use `max_iter=0`, add mocks,
or cover custom links.

## NB-OFFSET-002: Initializer documentation contract

Verify semantically that `_AbstractInit` documents an offset-free returned predictor and that public `irls`
documentation states its `eta` excludes the separately supplied offset. Do not assert literal docstring text.

Run:

```bash
uv run --extra docs zensical build --strict --clean
git diff HEAD -- src/jaxqtl/infer/_glm.py src/jaxqtl/infer/_optimize.py
```

The documentation build must exit successfully, and the two contracts must agree without signature changes.

## NB-OFFSET-003: Scope-preservation regression

Run:

```bash
uv run pytest -p no:capture tests/test_lm_glm.py tests/test_optimize.py -q
uv run pytest -p no:capture -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run ty check src tests scripts
git diff --check HEAD
```

Every command must exit successfully. Diff inspection must confirm that the complete preliminary Poisson predictor is
still used for the moment and dispersion calculations, only the returned initializer predictor changes to
`complete_eta - offset`, and `irls` executable logic is unchanged.

Item 2 excludes new JIT, `vmap`, gradient, solver-matrix, nonfinite, convergence-boundary, hypothesis-test, custom-link,
and offset-shape-validation coverage. Those broader checks remain item 3.
