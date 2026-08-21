# Negative-binomial initializer offset implementation plan

**Goal:** Ensure the negative-binomial warm start applies a supplied offset exactly once.

**Architecture:** Preserve the existing IRLS contract: initializers return an offset-free predictor and IRLS adds the
offset when constructing its initial state. Keep the complete preliminary Poisson predictor inside `_NBInit` for the
moment estimate and negative-binomial dispersion fit, then remove the offset only at the initializer-to-IRLS handoff.

**Tech Stack:** Python, JAX, Equinox, NumPy, pytest.

**Scope:** One phase covering item 2 only.

**Codebase verified:** 2026-08-21.

## Review Profile

`inference`

---

## Acceptance Criteria Coverage

This phase implements and tests:

### negative-binomial-offset.AC1: Single-offset warm-start semantics

- **negative-binomial-offset.AC1.1 Success:** With a nonzero scalar or vector offset, adding the offset once to the
  predictor returned by `_NBInit` reconstructs the complete preliminary Poisson predictor.
- **negative-binomial-offset.AC1.2 Success:** The complete preliminary Poisson predictor remains the input to the
  moment estimate and negative-binomial dispersion initialization.

### negative-binomial-offset.AC2: Initializer contract documentation

- **negative-binomial-offset.AC2.1 Success:** The initializer and public `irls` documentation state that the initial
  predictor excludes the separately supplied offset.

### negative-binomial-offset.AC3: Scope preservation

- **negative-binomial-offset.AC3.1 Success:** The change does not alter custom-link initialization, the IRLS update
  rule, hypothesis tests, solvers, or offset-shape validation.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->
<!-- START_TASK_1 -->
### Task 1: Add the negative-binomial initializer offset regression

**Verifies:** negative-binomial-offset.AC1.1, negative-binomial-offset.AC1.2

**Files:**

- Modify: `tests/test_lm_glm.py` (unit regression)

**Implementation:**

Add a deterministic scalar/vector-parameterized test near the existing Poisson GLM tests. Construct a default-log-link
negative-binomial initializer and the equivalent preliminary Poisson model using the same design, response, offset,
solver, and optimizer settings. Assert that adding the supplied offset once to the initializer result reconstructs the
complete Poisson `eta`. Independently reproduce the existing moment estimate and dispersion fit from that complete
predictor, including the existing `nan_to_num` fallback, and assert that the initializer returns the same dispersion.
Do not assert exact iteration counts or use `max_iter=0`, because those would couple the test to unrelated loop-boundary
behavior.

**Testing:**

- AC1.1: Both scalar and vector cases must expose the current extra-offset discrepancy.
- AC1.2: The returned dispersion must match an independent calculation that uses the real preliminary Poisson fit's
  complete predictor for both the moment estimate and `NegativeBinomial.estimate_dispersion`, without mocks.

**Verification:**

Run: `uv run pytest -p no:capture tests/test_lm_glm.py::test_negative_binomial_initializer_returns_predictor_without_offset -q`

Expected before production changes: two assertion failures whose differences equal the supplied offsets.

<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Correct and document the initializer handoff

**Verifies:** negative-binomial-offset.AC1.1, negative-binomial-offset.AC1.2,
negative-binomial-offset.AC2.1, negative-binomial-offset.AC3.1

**Files:**

- Modify: `src/jaxqtl/infer/_glm.py:50-118`
- Modify: `src/jaxqtl/infer/_optimize.py:48`

**Implementation:**

In `_NBInit.init`, name the preliminary Poisson result's `eta` as the complete predictor and continue using it for the
moment estimate and `NegativeBinomial.estimate_dispersion`. Return that complete predictor minus `offset` so the
unchanged IRLS initialization adds the fixed offset once. Document the offset-free return contract on `_AbstractInit`
and clarify the exported `irls` parameter documentation. Add the required `# pattern: Functional Core` classification
to `_optimize.py`; otherwise add only comments that explain offset ownership. Do not alter custom links, IRLS updates,
solvers, hypothesis paths, or shape validation.

**Testing:**

- AC1.1: The Task 1 regression becomes green for both offset forms.
- AC1.2: Existing dispersion initialization continues to receive complete `eta`.
- AC2.1: The internal and public docstrings describe the same ownership rule.
- AC3.1: Review confirms that the diff is confined to the initializer handoff and documentation.

**Verification:**

Run: `uv run pytest -p no:capture tests/test_lm_glm.py::test_negative_binomial_initializer_returns_predictor_without_offset -q`

Expected after production changes: two tests pass.

<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Verify item 2 and review its diff

**Verifies:** All item-2 acceptance criteria.

**Files:**

- Verify: `src/jaxqtl/infer/_glm.py`
- Verify: `src/jaxqtl/infer/_optimize.py`
- Verify: `tests/test_lm_glm.py`

**Implementation:**

Run the focused inference tests, complete project checks, and baseline plus numerics and inference reviews. Preserve the
pre-existing untracked `state.md`. Stop for user discussion before committing or beginning item 3.

**Verification:**

- `uv run pytest -p no:capture tests/test_lm_glm.py tests/test_optimize.py -q`
- `uv run pytest -p no:capture -q`
- `uv run ruff check src tests scripts`
- `uv run ruff format --check src tests scripts`
- `uv run ty check src tests scripts`
- `uv run --extra docs zensical build --strict --clean`
- `git diff --check HEAD`
- `git status --short`

Expected: all checks pass; only item-2 files and the plan artifact are changed, while `state.md` remains untouched.

<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->
