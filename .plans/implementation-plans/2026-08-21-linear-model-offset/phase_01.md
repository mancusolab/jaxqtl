# Linear model offset semantics implementation plan

**Goal:** Correct `LinearModel.fit` fitted-state and Huber-White covariance semantics for nonzero offsets.

**Architecture:** Keep the specialized Gaussian identity-link calculation inside `LinearModel.fit`. Fit coefficients
against `y - offset`, then define the complete linear predictor and fitted mean with the offset included. Do not add a
shared helper or modify hypothesis-test, GLM-initialization, IRLS, or solver code.

**Tech stack:** Python, JAX, Equinox, NumPy, statsmodels, pytest.

**Scope:** One phase covering item 1 only.

**Codebase verified:** 2026-08-21.

---

## Acceptance criteria coverage

### linear-model-offset.AC1: Complete fitted-state semantics

- **linear-model-offset.AC1.1 Success:** With a nonzero scalar or vector offset, `eta` equals `X @ beta + offset`.
- **linear-model-offset.AC1.2 Success:** For the enforced Gaussian identity link, `mu` equals `eta`.
- **linear-model-offset.AC1.3 Success:** Residuals equal `y - mu`, and dispersion remains RSS divided by residual DOF.

### linear-model-offset.AC2: Huber-White covariance

- **linear-model-offset.AC2.1 Success:** With scalar or vector offsets, the complete covariance matrix matches
  statsmodels HC0 covariance for OLS fitted to `y - offset`.

### linear-model-offset.AC3: Public result documentation

- **linear-model-offset.AC3.1 Success:** `ModelResult` documents `eta`, `mu`, and `resid` consistently with their
  returned values.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->
<!-- START_TASK_1 -->
### Task 1: Add the direct linear-model offset regression test

**Files:**

- Modify: `tests/test_lm_glm.py`

**Steps:**

1. Add a scalar/vector-parameterized test after `test_linear_model_matches_statsmodels`.
2. Generate an intercept-bearing design and heteroskedastic response with a nonzero offset.
3. Fit statsmodels OLS to `y - offset` and compute HC0 with `cov_white_simple(..., use_correction=False)`.
4. Assert coefficient, `eta`, `mu`, residual, dispersion, and complete covariance behavior.
5. Run the isolated test with `uv run pytest -p no:capture` and confirm it fails on `eta`, `mu`, and covariance before
   modifying production code.

**Verifies:** AC1.1, AC1.2, AC1.3, AC2.1.

<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Correct fitted-state construction and documentation

**Files:**

- Modify: `src/jaxqtl/infer/_glm.py`

**Steps:**

1. Define `eta` as `X @ beta + offset`.
2. Define `mu = eta` and document why this follows from the enforced Gaussian identity link.
3. Define residuals as `y - mu`.
4. Document the complete-linear-predictor, fitted-mean, and working-residual contracts on `ModelResult`.
5. Remove the conflicting residual field comment.

**Verifies:** AC1.1, AC1.2, AC1.3, AC2.1, AC3.1.

<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Verify item 1 and review its diff

**Files:**

- Verify: `src/jaxqtl/infer/_glm.py`
- Verify: `tests/test_lm_glm.py`

**Steps:**

1. Rerun the isolated regression test and confirm it passes.
2. Run `tests/test_lm_glm.py`, then the full test suite with the required `-p no:capture` option.
3. Run Ruff lint/format checks, type checking, strict documentation build, and `git diff --check`.
4. Review the item-1 diff before any commit or item-2 work.

**Verifies:** All item-1 acceptance criteria.

<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->
