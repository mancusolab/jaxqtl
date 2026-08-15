# Cis-by-state Phase 4 Implementation Plan

**Goal:** Compute the standardized-genotype cis-by-state statistic and nuisance-efficient donor contributions with fixed-size blocked operations.

**Architecture:** Add one cohesive functional-core module in `jaxqtl.hypothesis` that owns fixed-cohort genotype standardization, whole-kernel scaling, structured score algebra, nuisance projection, and a private matrix-free donor-contribution operator. Reuse Phase 3 grouped centering and the exact fitted nuisance design; never adapt the existing dense per-variant hypothesis hierarchy or materialize conceptual interaction/contribution matrices.

**Tech Stack:** Python 3.11+, JAX 0.10+, Equinox, Lineax 0.1+, NumPy, pytest, Ruff, ty

**Scope:** 7 phases from the original design; this file implements phase 4 of 7.

**Codebase verified:** 2026-08-14 16:22 PDT

---

## Review Profile

- Profile: `inference`
- Why this profile: This phase implements the nuisance-efficient score and matrix-free donor contribution operator, requiring joint numerical and inference-algorithm review.

## Acceptance Criteria Coverage

This phase implements and tests:

### cis-state.AC3: Omnibus score

- **cis-state.AC3.1 Success/Edge:** Genotypes are standardized; invalid or monomorphic variants are removed and reported.
- **cis-state.AC3.2 Contract:** Public APIs expose neither variant nor state weights.
- **cis-state.AC3.3 Success:** Structured `U` and `Q` equal explicit interaction-design calculations.
- **cis-state.AC3.4 Success:** Efficient donor contributions match explicit nuisance projection and sum to the efficient score.
- **cis-state.AC3.5 Success:** Results are invariant to valid variant-block choices.
- **cis-state.AC3.6 Scale:** Production paths never construct prohibited dense matrices.

---

<!-- START_TASK_1 -->
### Task 1: Standardize genotypes and fix whole-kernel scales

**Verifies:** cis-state.AC3.1, cis-state.AC3.2

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/_cluster_score.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/__init__.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_cluster_score.py` (unit)

**Implementation:**
- Start `_cluster_score.py` with `# pattern: Functional Core`; add `# pattern: Imperative Shell` to the import-only `hypothesis/__init__.py` before modifying exports.
- Keep the full Phase 4 responsibility in this module: private genotype/kernel helpers, structured scores, nuisance projection, contribution operator, immutable result/diagnostics, and `compute_cis_state_score`. Do not split standardization, operator, result, or validation leaf modules; Phase 5's stochastic calibration will be a separate cohesive module.
- Define `compute_cis_state_score(G_raw, S, donor_index, null_result, *, variant_block_size, input_variant_mask=None)` with no variant/state weights and no caller-selectable standardization cohort. Direct array callers omit the mask and receive an all-true mask; the Phase 6 bucketed workflow supplies a fixed-capacity mask for host-padded columns. Require a successful Phase 3 result and validate all fixed aligned dimensions at the host boundary.
- Require x64 to remain enabled and the successful null result to record float64. Promote accepted real `G_raw`/`S` inputs to float64 at the public host boundary before launching the float64-only internal score kernel; reject bool/complex inputs, disabled x64, or a non-float64 null result rather than silently changing the score scale.
- Standardize each input-valid genotype column once over the complete aligned donor cohort before any gene-specific fit/direct-state-informative masks: population mean and RMS (`ddof=0`), then `(G_raw - mean) / scale`. Drop/report a scientific variant if any aligned dosage is nonfinite or `scale <= sqrt(eps(dtype)) * max(1, max(abs(G_raw_j)))`; never restandardize a globally valid column on a per-gene donor subset. Treat input-mask-false padding as neutral capacity rather than a monomorphic/drop diagnostic.
- Retain the natural unwhitened state factors. Compute and report `c_G = ||G_std||_F^2 / N` over all aligned donors and `c_S = ||S||_F^2 / M` over all aligned selected cells, require both positive finite, and use `G_star = G_std / sqrt(c_G)` and `S_star = S / sqrt(c_S)`. Do not replace measured `c_G` by `p`, standardize state columns, or expose public weights.
- Colocate immutable `CisStateScoreResult`/diagnostics. Return original-length valid-variant mask, original-length means/scales with `NaN` for invalid columns, input/valid/drop counts and reasons, kernel scales, aligned/fit/direct-state-informative donor counts, block size, dtype/backend, status, and later score/projection diagnostics. Return a non-success/no-statistic result when no valid variant remains.

**Testing:**
- Write public standardization/contract tests first and confirm they fail before implementation.
- Verify exact population means/scales, zero means/unit RMS, nonfinite and numerically monomorphic drops, original-index diagnostics, all-invalid failure, and invariance of standardized columns to gene-specific fit masks.
- Verify natural state columns remain unchanged except for one global kernel factor; actual mean-diagonal genotype/state scales and normalized kernels match explicit dense calculations.
- In isolated boundary tests, require disabled x64 and mismatched null dtype to fail before score construction, float32 public inputs to be promoted deterministically to float64, and direct internal float32 launches to be rejected.
- Inspect the public function signature and exports to prove variant/state weight arguments and arbitrary covariance objects are absent.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cluster_score.py -x`
- Expected red: standardization/kernel/public-contract cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cluster_score.py`
- Expected green: the initial genotype and kernel tests pass.

**Commit:** `feat: standardize cis-state kernels`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Compute structured raw and nuisance-efficient scores

**Verifies:** cis-state.AC3.3, cis-state.AC3.4

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/_cluster_score.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/__init__.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cluster_score.py` (unit)

**Implementation:**
- Replace masked Phase 3 `NaN` outputs with finite zeros under `fit_cell_mask`. With final NB weights, reuse grouped weighted centering to form `X0` from `null_result.nuisance_design` and `S0` from normalized `S_star`; do not rebuild the nuisance design from donor covariates.
- Compute allowed small donor quantities by fixed segment sums: `A_raw,i = sum_c q_c S_star,c` and `A_i = sum_c q_c S0_c`, both `(N, r)`; nuisance score `b_i` `(N, d)`; donor state-nuisance information `J_SX,i` `(N, r, d)`; and global `J_XX` `(d, d)`. Use centered `A_i` in every efficient donor contribution and retain `A_raw,i` only for the explicit uncentered-score/convergence diagnostic. Only zero-total fit masks remove cells; one-cell/no-state-variation donors retain mathematically required nuisance-projection contributions and their separate diagnostics.
- Reuse the Phase 3 rank tolerance to require full-rank SPD `J_XX`. Before tagging it positive definite, require the overflow-safe equivalent `cond(J_XX) <= null_result.config.cholesky_accuracy_budget/eps(dtype)`; otherwise fail with `ill_conditioned_nuisance_information`. Initialize one `lineax.Cholesky` state and solve all `J_XX a_i = b_i` columns with `jax.vmap(..., state=shared_state, throw=False)`. Aggregate `Solution.result == lx.RESULTS.successful` on device and require every scaled backward error `||J_XX a_i-b_i||/(||J_XX||*||a_i||+||b_i||+tiny) <= normal_equation_rtol`; return solve status/residual/condition diagnostics to the host and never invert `J_XX`.
- For each valid genotype block, compute `J_HX,b = sum_i G_star[i,b] outer J_SX,i` with shape `(b, r, d)`, raw donor contribution `u_i,b = G_star[i,b,None] * A_i`, and efficient contribution `t_i,b = u_i,b - reshape(J_HX,b @ a_i, (b, r))` using variant-major/state-minor C-order flattening.
- Return both conceptual raw score `U_raw = G_star.T @ sum_i(q_i S_star_i)` for explicit-`H` validation and efficient score `U_eff = sum_i t_i`. Define the reported statistic as `Q = ||U_eff||_F^2`, record the raw/efficient discrepancy and score-equation residuals, and fail closed if the null result, information solve, or score arrays are non-success/nonfinite.
- Do not add a dispersion projection: Phase 3 tests expected NB2 mean-dispersion cross-information zero and records the observed-information caveat.

**Testing:**
- Add explicit small interaction-design and nuisance-projection tests first and observe failure before implementation.
- Build test-only `H` with variant-major/state-minor ordering and verify `vec_C(U_raw) = H.T @ q` plus its globally normalized quadratic-form identity.
- Build test-only donor dummies, `H0`, `X0`, `J_HX`, `J_XX`, and every `t_i`; compare blocked information, raw/efficient scores, nuisance projection, `Q`, and `sum_i t_i = vec_C(U_eff)`.
- Verify exact donor score equations at convergence and diagnostic discrepancies for intentionally perturbed fits; exercise one-cell/no-state-variation donors with varying cell covariates to ensure contributions are not naively zeroed.
- Parameterize unsuccessful/null-shape/rank/Lineax/nonfinite failures and verify they cannot return a finite statistic. Include algebraically full-rank but ill-conditioned `J_XX` cases on both sides of the `cond*eps` budget and compare accepted solves with a dense SVD oracle.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cluster_score.py -x`
- Expected red: structured/efficient score cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cluster_score.py`
- Expected green: genotype, raw-score, and nuisance-efficient score tests pass.

**Commit:** `feat: compute efficient cis-state score`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Add the blocked donor-contribution operator

**Verifies:** cis-state.AC3.4, cis-state.AC3.5, cis-state.AC3.6

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/_cluster_score.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cluster_score.py` (unit)

**Implementation:**
- Add a private immutable `_ContributionOperator` holding only normalized genotype/state score ingredients, `a_i`, `J_SX`, fixed-capacity validity masks, dimensions, and static variant block size. It must preserve the capacity axis instead of boolean-compacting valid variants. Phase 5 may import the private constructor/operator, but `jaxqtl.hypothesis` must not export it.
- Provide a private `_build_score_state(...) -> (CisStateScoreResult, _ContributionOperator)` used by the public score wrapper and later calibration. The public wrapper returns only the backend-neutral result; the Phase 5 internal path consumes the tuple once so it neither exposes the operator nor rebuilds the score state.
- Represent conceptual `T` with donor rows and variant-major/state-minor columns without storing `N x (p*r)`. Implement block-local contribution construction plus:
  - `matvec(x)` returning `T @ x`;
  - `rmatvec(v)` returning `T.T @ v` only when a feature vector is explicitly requested;
  - `rmatvec_squared_norm(v)` accumulating `||T.T @ v||^2` without allocating `p*r`;
  - `gram_matvec(v)` accumulating `T @ (T.T @ v)` for the Phase 5 donor-space operator;
  - `exact_trace()` accumulating `sum_b ||T_b||_F^2 = trace(T T.T)`.
- Host-pad genotype columns to a multiple of the fixed block size with finite zeros and a fixed-size valid-column mask before JIT. Use static-size slices/scans; never assume `lax.dynamic_slice` returns a short final block, since an overrun shifts the start backward. Mask padded calculations and outputs without boolean compaction.
- Permit the public score result to store `p x r` raw/efficient scores, but prohibit production allocations shaped `M x p`, `M x (p*r)`, `N x p x r`, `N x (p*r)`, or unbounded `p x r x d`. A bounded `N x block_size x r` temporary is allowed.
- Accumulate the reported statistic through the same blocked contribution path used by calibration, and keep block-size choice out of numerical meaning.

**Testing:**
- Add operator/property tests first and confirm behavioral failure.
- Against a tiny explicit `T`, verify matvec/rmatvec, adjoint inner products, squared norm, donor Gram action, exact first trace, contribution sums, and observed statistic.
- Parameterize block sizes `1`, exact divisor, nondivisor, and `>= p`; verify identical scores/statistics/operator actions, correct padded-final-block masking, and eager/JIT equality. Supply an explicit larger fixed-capacity `input_variant_mask` and verify padding changes neither scientific input/drop counts nor results.
- Add an allocation-shape recorder and sentinel inputs that fail on prohibited expansion. Verify the largest interaction temporary is bounded by configured block size and no final-block recompilation is needed within a fixed padded bucket.
- Use bounded Hypothesis equivalence cases for small valid shapes and block sizes; keep rank/failure cases as explicit examples.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cluster_score.py -x`
- Expected red: contribution-operator and block-invariance cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cluster_score.py`
- Expected green: all Phase 4 score/operator tests pass.

**Commit:** `feat: add blocked donor score operator`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Run Phase 4 regression and quality gates

**Verifies:** cis-state.AC3.1, cis-state.AC3.2, cis-state.AC3.3, cis-state.AC3.4, cis-state.AC3.5, cis-state.AC3.6

**Files:**
- Verify only; no planned file changes.

**Implementation:**
- Confirm `hypothesis/_score.py` and existing per-variant public abstractions remain unchanged and the new API exports neither private contribution operators nor feature weights.
- Audit JIT paths for dynamic compaction, unpadded short blocks, repeated Lineax factorizations, unstable result structure, and prohibited allocation shapes.
- Review source classifications and keep `/Users/nicholas/Projects/jaxqtl/state.md` untracked/unstaged.

**Verification:**
- Run: `uv run pytest -p no:capture tests/test_cluster_score.py tests/test_cell_nb.py tests/test_lm_glm.py tests/test_optimize.py`
- Expected: score, absorbed-NB, and adjacent regressions pass.
- Run: `uv run pytest -p no:capture`
- Expected: the complete suite passes with no new warnings or regressions.
- Run: `uv run ruff check src tests scripts`
- Expected: no lint errors.
- Run: `uv run ruff format --check src tests scripts`
- Expected: all files already formatted.
- Run: `uv run ty check src tests scripts`
- Expected: no type errors.
- Run: `git diff --check`
- Expected: no whitespace errors.
- Verifies: cis-state.AC3.1, cis-state.AC3.2, cis-state.AC3.3, cis-state.AC3.4, cis-state.AC3.5, cis-state.AC3.6.

**Commit:** None unless verification finds a required fix; repeat the owning task's tests and make a narrow correction commit.
<!-- END_TASK_4 -->
