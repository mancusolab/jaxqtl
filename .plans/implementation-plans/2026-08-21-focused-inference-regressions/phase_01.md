# Focused inference regression implementation plan

**Goal:** Add focused eager/JIT integration coverage for the linear/Wald, negative-binomial, and Score contracts changed
or exposed by the cleanup work.

**Architecture:** Extend the cohesive inference test modules with independent numerical oracles and public-boundary
parity checks. Use a self-executing Imperative Shell test module for the float32 negative-binomial contract because
JAX precision configuration is process-global. Keep this phase tests-only; if a regression fails, stop for diagnosis
instead of modifying production code or broadening the matrix.

**Tech Stack:** Python, JAX, Equinox, NumPy, pytest.

**Scope:** One phase covering item 3 only.

**Codebase verified:** 2026-08-21.

## Review Profile

`inference`

---

## Acceptance Criteria Coverage

This phase implements and tests:

### focused-inference-regressions.AC1: Gaussian Wald integration

- **focused-inference-regressions.AC1.1 Success:** Scalar and vector offsets with Fisher and Huber covariance match
  explicit full-model Gaussian fits for `beta`, `se`, `z`, and `p`.
- **focused-inference-regressions.AC1.2 Success:** Outer-JIT Wald results match eager results for all four offset and
  covariance cases.

### focused-inference-regressions.AC2: Negative-binomial integration

- **focused-inference-regressions.AC2.1 Success:** Scalar and vector negative-binomial initializer results agree under
  eager and outer-JIT execution while preserving the complete-Poisson-predictor dispersion oracle.
- **focused-inference-regressions.AC2.2 Success:** A public default-log-link negative-binomial fit with a nonconstant
  vector offset converges, returns a valid fitted state, satisfies its predictor/mean/residual identities, and agrees
  under eager and outer-JIT execution at a tolerance supported by float32.
- **focused-inference-regressions.AC2.3 Success:** The public negative-binomial fit contract is repeated in a fresh
  process with JAX x64 disabled, preventing module-level x64 configuration from masking float32 behavior.

### focused-inference-regressions.AC3: Gaussian Score integration

- **focused-inference-regressions.AC3.1 Success:** A Gaussian identity-link Score test with a vector offset matches an
  independent closed-form NumPy efficient-score calculation.
- **focused-inference-regressions.AC3.2 Success:** Outer-JIT Score results match eager results for every returned field.

### focused-inference-regressions.AC4: Scope preservation

- **focused-inference-regressions.AC4.1 Success:** The phase changes tests and plan artifacts only; it does not add SPA,
  custom-link, broad solver-matrix, `vmap`, AD, shape-validation, or convergence-boundary work.

---

<!-- START_SUBCOMPONENT_A (tasks 1-4) -->
<!-- START_TASK_1 -->
### Task 1: Consolidate Gaussian Wald offset and JIT coverage

**Verifies:** focused-inference-regressions.AC1.1, focused-inference-regressions.AC1.2

**Files:**

- Modify: `tests/test_hypothesis.py`

**Implementation:**

Merge the existing Gaussian Wald offset and outer-JIT regressions into one test parameterized over scalar/vector offset
and `FisherInfoError`/`HuberError`, using the default Cholesky solver. Compare eager results with explicit full-model
fits for every genotype, and compare outer-JIT results with eager results. Retain the separate zero-offset all-solver
test instead of multiplying this matrix across solvers.

**Testing:**

- AC1.1: Assert `(m,)` result shapes and full-model agreement for `beta`, `se`, `z`, and `p` using `rtol=2e-4` and
  `atol=2e-5`.
- AC1.2: Assert eager/JIT agreement for `beta`, `se`, `z`, `p`, and `disp` using `rtol=1e-5` and `atol=1e-5`, with
  exact iteration and convergence metadata.

**Verification:**

Run: `uv run pytest -p no:capture tests/test_hypothesis.py::test_gaussian_wald_with_offset_matches_full_model_and_jit -q`

Expected: four cases pass. Any failure blocks production edits and triggers diagnosis.

<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Expand negative-binomial eager/JIT coverage

**Verifies:** focused-inference-regressions.AC2.1, focused-inference-regressions.AC2.2

**Files:**

- Modify: `tests/test_lm_glm.py`
- Create: `tests/test_precision_modes.py`

**Implementation:**

Extend the existing scalar/vector initializer test with outer-JIT parity assertions while retaining its independent
complete-predictor dispersion oracle and offset reconstruction. Add one public default-log-link negative-binomial fit
using the verified deterministic vector-offset fixture (seed 18, `n=160`, `max_iter=200`, `tol=1e-4`). Compare eager
and outer-JIT fitted states. Repeat that public fit in a fresh child process with `JAX_ENABLE_X64=0`; keep the parent
test module free of JAX imports so collection cannot initialize or reconfigure the child process.

**Testing:**

- AC2.1: Assert eager/JIT predictor and dispersion parity without requiring bitwise dispersion equality.
- AC2.2: Assert both fits converge; dispersion is positive and finite; all fitted numerical fields are finite;
  `eta == X @ beta + offset`; `mu == exp(eta)`; log-link NB residuals equal `(y - mu) / mu`; all numerical fields agree
  under eager/JIT within `rtol=1e-5`, `atol=1e-5`; iteration and convergence metadata agree exactly.
- AC2.3: Assert the child process is genuinely in x32 mode and repeats the convergence, identity, and eager/JIT parity
  checks, comparing fitted fields with a separate cross-backend parity tolerance (`rtol=1e-3`, `atol=1e-4`) to
  accommodate observed float32 reduction and fusion-order variation. This comparison tolerance is independent of the
  objective-change convergence tolerance. Run the self-executing test file with `sys.executable`, a resolved path, an
  explicit repository working directory, captured output, and a finite timeout.
- Do not assert exact iteration count, generating-parameter recovery, or statsmodels equivalence.

**Verification:**

Run: `uv run pytest -p no:capture tests/test_lm_glm.py::test_negative_binomial_initializer_returns_predictor_without_offset tests/test_lm_glm.py::test_negative_binomial_fit_with_offset_matches_jit tests/test_precision_modes.py::test_negative_binomial_fit_with_offset_in_x32_process -q`

Expected: four cases pass. Any failure blocks production edits and triggers diagnosis.

<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Add the direct Gaussian Score oracle

**Verifies:** focused-inference-regressions.AC3.1, focused-inference-regressions.AC3.2

**Files:**

- Modify: `tests/test_hypothesis.py`

**Implementation:**

Add one deterministic `LinearModel`/`FisherInfoError` Score regression with a nonconstant vector offset correlated with
one genotype. Independently residualize `y - offset` and every genotype against the covariates in NumPy, then calculate
the null dispersion, efficient score, information, effect estimate, standard error, signed statistic, and two-sided
Normal p-value. Compare the public eager result to this oracle and outer-JIT to eager. Do not use Wald as the oracle.

**Testing:**

- AC3.1: Assert result shapes, finiteness, convergence metadata, and oracle agreement for `beta`, `se`, `z`, `p`, and
  `disp` using `rtol=2e-4`, `atol=2e-5`.
- AC3.2: Assert every Score result field agrees under eager/JIT, using `rtol=1e-5`, `atol=1e-5` for numerical fields and
  exact metadata comparison.

**Verification:**

Run: `uv run pytest -p no:capture tests/test_hypothesis.py::test_gaussian_score_with_offset_matches_closed_form_and_jit -q`

Expected: one case passes. Any failure blocks production edits and triggers diagnosis.

<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Verify and review the focused regression phase

**Verifies:** All focused-inference-regressions acceptance criteria.

**Files:**

- Verify: `tests/test_hypothesis.py`
- Verify: `tests/test_lm_glm.py`
- Verify: `tests/test_precision_modes.py`

**Implementation:**

Run focused, full, and static verification followed by baseline, numerics, and inference review. Confirm the diff is
tests-only apart from plan artifacts and preserve the pre-existing untracked `state.md`. Stop for discussion before
committing or beginning any additional work.

**Verification:**

- Focused new-behavior selection: expected 9 parameterized cases.
- `uv run pytest -p no:capture tests/test_hypothesis.py tests/test_lm_glm.py -q`: expected 46 passed.
- `uv run pytest -p no:capture tests/test_precision_modes.py -q`: expected 1 passed.
- `uv run pytest -p no:capture -q`: expected 141 passed.
- `uv run ruff check src tests scripts`
- `uv run ruff format --check src tests scripts`
- `uv run ty check src tests scripts`
- `uv run --extra docs zensical build --strict --clean`
- `git diff --check HEAD`
- `git status --short`

Expected: all checks pass, no production file changes, and `state.md` remains untouched.

<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_A -->
