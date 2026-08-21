# Human test plan: negative-binomial initializer offset

## Preconditions

- Work from branch `cleanup` with item-2 changes uncommitted.
- Do not modify or stage the untracked `state.md`.
- Use project development and documentation dependencies through `uv`.

## Phase checks

| Step | Action | Expected result |
|---|---|---|
| 1 | Inspect `_NBInit.init` in `src/jaxqtl/infer/_glm.py`. | The preliminary Poisson result is `complete_eta`; both moment and dispersion calculations consume it. |
| 2 | Inspect the `_NBInit.init` return expression. | It returns `complete_eta - offset`. |
| 3 | Inspect the IRLS initial tuple in `src/jaxqtl/infer/_optimize.py`. | IRLS adds `offset` once through `eta + offset`; no update-loop expression changed. |
| 4 | Compare the `_AbstractInit` and `irls` docstrings. | Both assign offset ownership to IRLS and define the initializer predictor as offset-free. |
| 5 | Inspect the regression in `tests/test_lm_glm.py`. | It covers scalar/vector offsets, verifies dispersion before predictor reconstruction, and uses no mocks or iteration-count assertions. |
| 6 | Inspect the scoped diff. | No custom-link, hypothesis, solver, offset-validation, or unrelated optimizer behavior changed. |
| 7 | Run `git status --short`. | Only item-2 files and plan artifacts changed; `state.md` remains untracked. |

## End-to-end scenarios

| Scenario | Expected result |
|---|---|
| Scalar offset ownership | Adding the scalar offset once to the initializer result reconstructs complete Poisson `eta`. |
| Vector offset ownership | Adding the vector offset once elementwise reconstructs complete Poisson `eta`. |
| Dispersion preservation | Initializer dispersion matches the independent calculation using complete Poisson `eta` and the existing NaN fallback. |
| Documentation contract | Strict docs build succeeds and both docstrings describe one consistent ownership rule. |
| Regression containment | Focused and full suites pass without introducing item-3 behavior or coverage. |

## Traceability

| Acceptance criterion | Automated evidence | Human check |
|---|---|---|
| negative-binomial-offset.AC1.1 | Scalar/vector regression cases | Phase checks 1–3 |
| negative-binomial-offset.AC1.2 | Independent dispersion calculation | Phase checks 1 and 5 |
| negative-binomial-offset.AC2.1 | Strict documentation build | Phase check 4 |
| negative-binomial-offset.AC3.1 | Focused/full suites and static checks | Phase checks 3, 6, and 7 |
