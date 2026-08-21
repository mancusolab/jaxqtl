# Test requirements: focused inference regressions

**Status:** Pre-execution requirements

**Scope:** Item 3 only; tests and plan artifacts only.

## Traceability matrix

| Test ID | Acceptance criteria | Expected test | Cases |
|---|---|---|---:|
| FI-WALD-001 | AC1.1, AC1.2 | `tests/test_hypothesis.py::test_gaussian_wald_with_offset_matches_full_model_and_jit` | 4 |
| FI-NB-INIT-001 | AC2.1 | `tests/test_lm_glm.py::test_negative_binomial_initializer_returns_predictor_without_offset` | 2 |
| FI-NB-FIT-001 | AC2.2 | `tests/test_lm_glm.py::test_negative_binomial_fit_with_offset_matches_jit` | 1 |
| FI-NB-X32-001 | AC2.3 | `tests/test_precision_modes.py::test_negative_binomial_fit_with_offset_in_x32_process` | 1 |
| FI-SCORE-001 | AC3.1, AC3.2 | `tests/test_hypothesis.py::test_gaussian_score_with_offset_matches_closed_form_and_jit` | 1 |
| FI-SCOPE-001 | AC4.1 | Diff and suite inspection | N/A |

The focused new-behavior selection contains nine pytest cases.

## FI-WALD-001

Parameterize scalar/vector offsets independently from `FisherInfoError()`/`HuberError()`. Use a well-conditioned
intercept-bearing design, multiple genotypes, nonzero effects, and heteroskedastic noise. Keep the existing zero-offset
all-solver regression separate.

For every genotype, fit the explicit full design with the same offset and covariance estimator. Assert `(m,)` shapes
and eager agreement for `beta`, `se`, `z`, and `p` using `rtol=2e-4`, `atol=2e-5`. Compare outer-JIT with eager for
all `TestResult` fields, using `rtol=1e-5`, `atol=1e-5` for numerical fields and exact metadata equality.

## FI-NB-INIT-001

Retain the existing scalar/vector fixture, independent complete-Poisson-predictor dispersion calculation, and
single-offset reconstruction. Add outer-JIT execution of `_NBInit.init`; compare predictor and dispersion with eager
using `rtol=1e-5`, `atol=1e-5`, and reconstruct complete Poisson `eta` from the JIT result. Do not assert exact
iterations, use `max_iter=0`, mock components, or add custom links.

## FI-NB-FIT-001

Use the approved seed-18, `n=160`, default-log-link, nonconstant-vector-offset fixture with `max_iter=200`, `tol=1e-4`.
Exercise public eager and outer-JIT `GeneralizedLinearModel.fit`.

For both results, require convergence, a positive finite dispersion, finite numerical fields, correct field shapes,
`eta == X @ beta + offset`, `mu == exp(eta)`, and `resid == (y - mu) / mu`. Compare every numerical `ModelResult` field
with `rtol=1e-5`, `atol=1e-5` and metadata exactly. Do not assert exact iteration count, parameter recovery, or
statsmodels equivalence.

## FI-NB-X32-001

Repeat FI-NB-FIT-001 in a fresh child process with `JAX_ENABLE_X64=0`, and assert inside the child that x64 is disabled
before running the fit. Use one self-executing `tests/test_precision_modes.py` module whose collection-time imports are
standard-library-only; this prevents the suite's module-level x64 configuration from contaminating the precision
contract and avoids a separate non-collected helper module. The parent test must report captured child output on
failure and use a finite timeout.

## FI-SCORE-001

Use `LinearModel`, Fisher information, an intercept-bearing design, multiple genotypes, and a nonconstant vector offset
correlated with a genotype. The oracle must not call Score/Wald or private hypothesis helpers.

Independently residualize `y - offset` and `G` against `X` using NumPy least squares. From the null residual degrees of
freedom calculate dispersion, efficient score, information, effect, standard error, signed statistic, and two-sided
Normal p-value using standard-library `math.erfc`. Assert shapes, finiteness, convergence, and eager oracle agreement
with `rtol=2e-4`, `atol=2e-5`. Compare every outer-JIT result field with eager, using `rtol=1e-5`, `atol=1e-5` for
numerical fields and exact metadata equality.

## FI-SCOPE-001

Only `tests/test_hypothesis.py`, `tests/test_lm_glm.py`, `tests/test_precision_modes.py`, and this item-3 plan directory
may change. Do not add production
edits, SPA, custom NB links, broad solver matrices, offset-shape validation, `vmap`, AD, CG weighted-GLM, nonfinite, or
convergence-boundary work. Preserve the existing zero-offset solver matrix and Huber rejection tests. Do not modify or
stage the pre-existing untracked `state.md`.

## Verification

```bash
uv run pytest -p no:capture tests/test_hypothesis.py::test_gaussian_wald_with_offset_matches_full_model_and_jit tests/test_lm_glm.py::test_negative_binomial_initializer_returns_predictor_without_offset tests/test_lm_glm.py::test_negative_binomial_fit_with_offset_matches_jit tests/test_precision_modes.py::test_negative_binomial_fit_with_offset_in_x32_process tests/test_hypothesis.py::test_gaussian_score_with_offset_matches_closed_form_and_jit -q
uv run pytest -p no:capture tests/test_hypothesis.py tests/test_lm_glm.py -q
uv run pytest -p no:capture tests/test_precision_modes.py -q
uv run pytest -p no:capture -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run ty check src tests scripts
uv run --extra docs zensical build --strict --clean
git diff --check HEAD
git status --short
```

Expected collection after implementation: nine selected cases, 46 original focused-module cases, one precision-mode
case, and 141 full-suite cases.
Counts must be reconciled with actual collection. Any failed new regression requires diagnosis before a production edit
or scope expansion.

## Human checks

- Confirm Wald and Score oracles are independent of the hypothesis implementation paths.
- Confirm offsets are nonzero, vector offsets are nonconstant, and the Score offset is genotype-correlated.
- Confirm outer JIT wraps public bound methods and compares stable result PyTrees.
- Confirm floating-point assertions use tolerances and exact equality is limited to metadata.
- Confirm no production or out-of-scope test file changed and `state.md` remains untouched.
