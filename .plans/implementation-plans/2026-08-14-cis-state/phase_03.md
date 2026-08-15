# Cis-by-state Phase 3 Implementation Plan

**Goal:** Fit the cell-level NB2 null with absorbed donor fixed effects and required donor-covariate-by-state nuisance adjustment.

**Architecture:** Add reusable fixed-shape grouped/FWL primitives and a separate specialized absorbed-NB controller inside `jaxqtl.infer`, reusing the existing family, link, likelihood, dispersion derivatives, and reduced solver interface without routing through the dense `GeneralizedLinearModel`. Keep per-gene cell/donor axes fixed with masks under JIT, return stable status/diagnostic structures, and fail closed before hypothesis testing.

**Tech Stack:** Python 3.11+, JAX 0.10+, Equinox, Lineax, NumPy, SciPy, Statsmodels 0.14+, pytest, Ruff, ty

**Scope:** 7 phases from the original design; this file implements phase 3 of 7.

**Codebase verified:** 2026-08-14 16:12 PDT

---

## Review Profile

- Profile: `inference`
- Why this profile: This phase implements the absorbed NB2 estimator, convergence controller, rank diagnostics, and inferential failure semantics without adding a public CLI or artifact boundary.

## Acceptance Criteria Coverage

This phase implements and tests:

### cis-state.AC2: Absorbed NB inference

- **cis-state.AC2.1 Failure:** `donor_covariates` is required; missing or incomplete values are rejected.
- **cis-state.AC2.2 Success:** Donor covariates enter as donor-covariate-by-state nuisance terms; donor-constant main effects and a global intercept are absent.
- **cis-state.AC2.3 Success:** Absorbed weighted least squares matches explicit donor-dummy weighted least squares.
- **cis-state.AC2.4 Success:** Absorbed NB fits match dense references for means, coefficients, dispersion, likelihood, weights, and score residuals.
- **cis-state.AC2.5 Edge:** Zero-total and otherwise uninformative donors are handled and reported according to the contract.
- **cis-state.AC2.6 Failure:** Nonconvergence, rank failure, or insufficient informative donors cannot produce a nominal p-value.

---

<!-- START_TASK_1 -->
### Task 1: Expose reusable NB2 likelihood and dispersion primitives

**Verifies:** Supporting numerical behavior for cis-state.AC2.4; the complete fit criterion is verified in Task 4.

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/distribution/_expfam.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_families.py` (unit)

**Implementation:**
- Add `# pattern: Functional Core` to `_expfam.py` before modifying it.
- Refactor `NegativeBinomial.negloglikelihood` to sum a new per-observation NB2 negative-log-likelihood primitive, preserving the existing public shape and NB2 parameterization while replacing cancellation-prone arithmetic. Use `log_mu=eta` directly for the log link and stable `log1p`/`logaddexp` expressions. Choose the gamma-ratio and mean-normalizer branches independently. For integer `y`, write gamma cancellation as `sum_{k=0}^{y-1} log1p(k*phi)` and evaluate up to order `16` as `sum_{n=1}^{16} (-1)^(n+1) * y * u_y^n * s_n(y) / n`, where `u_y=phi*y` and `s_n(y)=sum_{k=0}^{y-1}k^n/y^(n+1)` is evaluated by Horner form from the exact rational Bernoulli/Faulhaber coefficients (with `y=0/1` handled exactly), never raw `y^n`. Evaluate `log1p(phi*mu)/phi` through the analogous 16-term factored Horner polynomial `mu*sum_{n=0}^{16}(-u_mu)^n/(n+1)`, `u_mu=phi*mu`. For either scale factor `a` (`a=y` for gamma, `a=mu` for the mean term) and `u<1`, use conservative omitted-tail bounds `B0=a*u^17/(17*(1-u))`, `B1=a*u^17/(1-u)^2`, and `B2=17*a*u^17/(1-u)^3` for the value and first/second log-phi derivatives (the mean value's exact first denominator is `18`, so `17` is conservative). Use a series branch only when `u<=0.01` and all three bounds satisfy `numerics_atol + numerics_rtol*max(1,abs(partial_order))`, with fixed `numerics_atol=1e-10` and `numerics_rtol=1e-12`; otherwise use the corresponding general `gammaln` or `log1p` branch independently. Return numerical-domain failure if neither branch meets the finite/error contract. Apply `lax.cond`/mapped scalar branches so unstable inactive branches are not evaluated, and differentiate the selected stable formula for score/Hessian. The new absorbed controller applies fixed-shape masks to finite terms before summing instead of dynamically compacting cells.
- Expose the stable log-dispersion score/Hessian calculation as a reusable NB-specific method within the distribution module, differentiating the same branch-consistent likelihood rather than a cancellation-prone parallel formula. Keep `update_dispersion`/`estimate_dispersion` public signatures and established behavior unchanged for existing callers.
- Define the float64 predictor domain as `[log(finfo(float64).tiny)+2, log(finfo(float64).max)-2]`. A finite predictor outside it is an explicit numerical-domain failure in the new absorbed kernel; safe values may be substituted only inside an inactive/failure computation branch and may never produce success. Evaluate `log(1+phi*mu)` as `logaddexp(0,log(phi)+eta)`, `w=exp(eta-log_denom)`, and `q=y*exp(-log_denom)-w`; reject nonfinite working-response terms instead of clipping them.
- Do not duplicate the NB2 PMF, log link, variance, or dispersion derivatives in the new inference module.

**Testing:**
- Add tests first for the wished-for per-observation result and confirm the targeted test fails before implementation.
- Verify per-observation terms sum to the scalar likelihood across scalar/vector offsets and dispersions, match SciPy in the ordinary domain and an 80-digit `decimal` integer-count oracle near the Poisson boundary within the pinned `1e-10 + 1e-12*|oracle|` contract, remain finite on the supported predictor domain, and preserve ordinary-domain gradient values. Test each independent series/remainder switch at/below/above its boundary, including `y=1, mu=1e12, phi=1e-12` for gamma cancellation, `y=1, mu=1e196, phi=1e-200` for factored mean-series overflow resistance, and maximum accepted `y=2**53` at both sides of the error-bound decision. Include predictors at both supported limits and finite out-of-domain predictors yielding an explicit nonfinite-domain result rather than `NaN` contamination.
- Verify the reusable log-dispersion score/Hessian against high-accuracy finite differences/complex-step where supported and the high-precision oracle. Require continuity within pinned float64 tolerances across the branch point plus finite eager/JIT/JVP values.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_families.py -x`
- Expected red: the new per-observation/dispersion primitive tests fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_families.py`
- Expected green: all existing and new family tests pass.

**Commit:** `refactor: expose NB2 likelihood terms`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Implement grouped weighted centering and absorbed WLS

**Verifies:** cis-state.AC2.3

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/infer/_absorbed.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/infer/__init__.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_cell_nb.py` (unit)

**Implementation:**
- Start `_absorbed.py` with `# pattern: Functional Core`; add `# pattern: Imperative Shell` to the import-only `infer/__init__.py` before modifying its exports.
- Colocate private fixed-`num_segments` grouped sums/means/centering, rank diagnostics, immutable `AbsorbedWLSResult`, and public `absorbed_wls`; do not create `_segments.py`, `_fwl.py`, result-only, or validation-only modules.
- At the non-jitted `absorbed_wls` boundary require `jax.config.x64_enabled` before creating any JAX array; fail with an actionable host error when disabled. Reject bool/complex numerical inputs, promote accepted real floating inputs to `float64`, keep validated donor indices integer, and require every internal absorbed-WLS kernel floating array to be `float64`. Record resolved dtype/backend in diagnostics; do not rely on JAX's disabled-x64 truncation warning.
- Implement grouped reductions with `jax.ops.segment_sum` over a validated dense donor index and a static `num_donors`. Support unsorted repeated donor IDs, use finite zero-valued masks rather than traced boolean slicing, and return safe zeros plus explicit validity masks for groups with zero total weight.
- For each weighted solve, compute donor-weight sums, weighted donor means of `X` and `z`, centered `X0`/`z0`, and the exact FWL nuisance solution through a host-validated `QRSolve` or `CholeskySolve`. The new API deliberately rejects `CGSolve` and any other `AbstractLinearSolve` implementation because the existing solver contract discards convergence status and CG may throw under tracing. From the rank SVD record `cond_Xw=s_max/s_min`; before `CholeskySolve`, require the overflow-safe equivalent `cond_Xw <= sqrt(cholesky_accuracy_budget/eps(dtype))`, otherwise return `ill_conditioned_for_cholesky` and direct the caller to QR. After either supported solve, check coefficients for finiteness and require scaled backward error `||A@beta-b||_2 / max(||A||_2*||beta||_2+||b||_2,tiny) <= normal_equation_rtol`, where `A=X0.T@(w[:,None]*X0)` and `b=X0.T@(w*z0)`; return an array-valued failure code otherwise. Recover each donor effect uniquely as the weighted donor mean of `z - X beta`; do not impose sum-to-zero normalization because there is no global intercept.
- Diagnose the rank of `sqrt(w) * X0` with a reduced JAX SVD and explicit `rank_rtol` (default `max(M_active, p) * eps(dtype)`, configurable). Record rank, singular extrema, condition estimate, threshold, donor weight sums, and normal-equation residual. Use `lax.cond` to avoid executing an invalid solve under JIT and return a stable non-success result rather than raising inside traced code.
- Keep raw segment helpers private. The numerical kernel returns only arrays, an `int32` status code, and fixed-shape diagnostics. A non-jitted host wrapper performs `device_get`, maps the code to the exported status enum/actionable message, and builds `AbsorbedWLSResult`; export only that wrapper, result, and diagnostics from `jaxqtl.infer`.

**Testing:**
- Write dense donor-dummy WLS oracle tests first and confirm the public behavior fails before implementation.
- Compare nuisance coefficients, donor effects, fitted values, objective, weighted normal equations, and centered donor means against `[N donor dummies, X]` with no intercept over unsorted donor rows and heterogeneous weights.
- Verify vector/block grouped means, fixed-shape masks, zero-weight groups, one-cell groups, dense-index validation at the host boundary, structural degrees of freedom, collinear centered designs, configurable rank thresholds, and QR/Cholesky reduced solvers.
- Verify eager/JIT agreement and JVP/finite-difference behavior on the array-only full-rank absorbed-WLS kernel, then verify host code/message mapping separately. Test rejection of `CGSolve` and a custom solver at the host boundary plus finite/backward-error failure propagation from each supported solver. Include a nearly collinear but algebraically full-rank fixture: QR must match an SVD/lstsq oracle within tolerance while Cholesky is refused by the `cond_Xw^2*eps` gate rather than returning an inaccurate success. Do not claim arbitrary batching; only test identical fixed-shape inputs if a batch wrapper is exposed.
- In an isolated subprocess, disable x64 and require the public wrapper to fail before tracing. With x64 enabled, pass float32 host arrays and require explicit promotion plus float64 diagnostics/results; direct internal-kernel float32 invocation must be rejected by its host launch validator. Keep these global-config tests isolated so they cannot contaminate the rest of the suite.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cell_nb.py -x`
- Expected red: grouped/FWL cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cell_nb.py`
- Expected green: grouped and absorbed-WLS tests pass.

**Commit:** `feat: add grouped absorbed WLS`
<!-- END_TASK_2 -->

<!-- START_SUBCOMPONENT_A (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Validate the cell-level nuisance design and donor masks

**Verifies:** cis-state.AC2.1, cis-state.AC2.2, cis-state.AC2.5

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/infer/_cell_nb.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/infer/__init__.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cell_nb.py` (unit)

**Implementation:**
- Start `_cell_nb.py` with `# pattern: Functional Core`. Colocate immutable configuration, integer kernel status constants, host-visible status mapping, `AbsorbedNBDiagnostics`, `AbsorbedNBResult`, boundary validation/design construction, and the specialized kernel; do not add types, status, validation, or dispersion leaf modules. Runtime-dependent strings/enums never cross a JAX transformation: the kernel returns array status codes and fixed-shape arrays, while the host wrapper maps codes to enums/messages after `device_get`.
- Apply the same explicit-x64 boundary to `fit_absorbed_nb`: check x64 before JAX array creation, promote accepted real `y`, `S`, `offset`, donor covariates, and optional cell covariates to float64, preserve integer donor indices, and reject bool/complex or nonconforming inputs. The fixed-shape kernel accepts only float64 floating leaves, and the public result records `jax_enable_x64=true`, resolved backend, and dtype.
- Define the public signature as `fit_absorbed_nb(y, S, donor_index, offset, donor_covariates, *, covariates=None, min_informative_donors, solver, config=None)`. Keep `donor_covariates` a required positional argument and `min_informative_donors` a required positive keyword until Phase 7 evidence can justify a default.
- Define `config=None` as the immutable effective configuration `AbsorbedNBConfig(mean_maxiter=100, dispersion_maxiter=50, predictor_rtol=1e-7, coefficient_rtol=1e-7, delta_log_phi_atol=1e-6, phi_lower=1e-8, phi_upper=1e3, max_backtracks=20, rank_rtol=None, normal_equation_rtol=1e-8, cholesky_accuracy_budget=1e-8)`, where `rank_rtol=None` alone means the documented shape/dtype rule `max(M_active,d)*eps(dtype)`. A supplied config replaces this whole configuration; there is no fieldwise merge with hidden defaults. Return the fully resolved config in diagnostics/manifests and document every field/range.

| `fit_absorbed_nb` argument | Required/default contract | Validation and precedence |
| --- | --- | --- |
| y/S/donor_index/offset/donor_covariates | required positional arrays | Exact aligned shapes/domains; donor covariates nonempty and finite |
| covariates | `None` | A supplied finite `M x c_dim` array replaces absence |
| min_informative_donors | required positive keyword | No library-level scientific default |
| solver | required `QRSolve` or `CholeskySolve` instance | Reject CG/unknown solver before tracing |
| config | exact immutable configuration above | `None` resolves exactly; a supplied config replaces every field as one object |

- Require every fixed or initialized NB2 dispersion to be strictly positive and finite. The exact Poisson case is outside this log-dispersion API; a configured positive lower bound represents the near-Poisson numerical boundary and is never returned as inferential success when hit.
- Require `donor_covariates` shape `(N, d_D)` with `d_D >= 1`, complete finite rows for every aligned donor, and at least one nonconstant column. Standardize columns across the fixed aligned donor cohort, return the means/scales, and reject zero-scale columns; later active-donor collinearity remains a weighted-rank failure.
- Build nuisance columns in fixed documented order: state main effects `S`, optional cell covariates, then donor-covariate-major/state-minor `D[donor_index, :, None] * S[:, None, :]` flattened in C order. Add neither a global intercept nor donor-covariate main effects. Reject a supplied explicit constant cell-covariate column through the centered-rank contract rather than silently dropping it.
- Compute original-shape donor totals and cell counts with fixed segment reductions. Mark zero-total donors/cells out of the likelihood and fit; retain their original positions with `NaN` fit outputs instead of substituting a finite donor intercept. Separately mark direct-state-informative donors as positive-total donors with at least two cells and nonzero within-donor state variation, preserving reason codes and original donor mapping. This second mask is diagnostic/threshold information, not an instruction to zero later nuisance-projection contributions; only zero-total donors are removed from the fit cohort.
- Keep `fit_donor_mask`, `fit_cell_mask`, and `informative_donor_mask` distinct. Do not physically filter traced arrays or change their shape per gene.

**Testing:**
- Add boundary/design/mask tests first and confirm they fail before implementation.
- Verify exact nuisance column values/order, donor-covariate standardization metadata, no intercept/main-effect columns, optional cell covariates, and deterministic donor masks/reason counts.
- Parameterize missing argument behavior, exact `config=None` resolution, whole-config replacement, wrong/incomplete/nonfinite donor covariates, zero-scale donor covariates, incompatible dimensions, invalid counts/offset/donor indices, constant/collinear nuisance designs, zero-total donors, single-cell donors, and donors without within-donor state variation.
- Add isolated disabled-x64 direct-API failure tests, float32-to-float64 public-boundary promotion tests, internal float32 launch rejection, and resolved dtype/backend diagnostics for QR and Cholesky.
- Verify masking preserves original result shapes and uses finite safe intermediates under eager and JIT execution.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cell_nb.py -x`
- Expected red: design/mask cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cell_nb.py`
- Expected green: nuisance-design and donor-mask tests pass with the grouped tests.

**Commit:** `feat: build absorbed cell-NB design`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Fit the alternating absorbed NB2 null

**Verifies:** cis-state.AC2.4, cis-state.AC2.5, cis-state.AC2.6

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/infer/_cell_nb.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/infer/__init__.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cell_nb.py` (unit/integration)

**Implementation:**
- Initialize each positive-total donor intercept as `log(sum_i y)-logsumexp(offset_i)` using stable segment log-sum-exp, initialize nuisance coefficients to zero, and compute the NB initialization scale through guarded log-domain ratios over fit cells. Treat a nonfinite/nonpositive initial dispersion as an explicit initialization failure rather than replacing it silently.
- Implement a fixed-shape specialized JAX controller with nested bounded `lax.while_loop` or equivalent stable control flow. Conditional on dispersion, use `log_mu=eta`, `z = eta + y*exp(-eta) - 1 - offset`, `log_denom=logaddexp(0,log(phi)+eta)`, `w=exp(eta-log_denom)`, and Task 2 absorbed WLS; guard the working response/predictor domain and return a numerical status instead of clipping. Update log dispersion from Task 1's branch-consistent stable score/Hessian with bounded backtracking that must reduce masked exact NB2 negative log likelihood.
- Require convergence of the conditional mean fit and the outer dispersion loop using documented relative predictor/coefficient changes plus absolute `delta log(phi)`. Record inner/outer iterations, final changes, NLL, rank diagnostics, backtracking count, dispersion bounds, and first failure code.
- Treat mean/dispersion maximum iterations, invalid rank, nonfinite mean/weight/likelihood/derivative, indefinite or rejected Newton step, positive lower near-Poisson-boundary hit, upper-bound hit, and fewer than `min_informative_donors` as explicit non-success statuses. Do not accept `phi=0`, silently substitute Poisson, or return a clipped success.
- Return original-shape `eta`, `mu`, weights, the stably evaluated score residual `y*exp(-log_denom)-w`, donor effects, nuisance coefficients, the exact fixed-order `nuisance_design` used by the fit, dispersion, masks, standardization metadata, and stable diagnostics. The `M x d` nuisance design is an allowed small matrix and prevents later hypothesis code from reconstructing hidden Phase 3 internals. The private kernel result is an array-only fixed-structure PyTree; mask excluded-cell fit quantities to `NaN` only after finite internal computation. The public host wrapper adds the enum/message after `device_get` and is not itself promised to be jittable.
- Expose no p-value field. Downstream hypothesis code must require success plus the informative-donor threshold; a failed result cannot be interpreted as inferential output.
- Document/test expected mean-dispersion orthogonality: `d U_eta / d phi` has conditional expectation zero, while the observed mixed Hessian need not be zero. Do not project dispersion from Phase 4 expected-information scores.

**Testing:**
- Write fixed-dispersion and full-fit reference tests first and observe the expected behavioral failure.
- For fixed dispersion, compare against Statsmodels `GLM` with `NegativeBinomial(alpha=phi)`, a full donor-dummy matrix, the exact nuisance design, no intercept, and the same offset. Statsmodels GLM alpha is fixed; do not use its discrete NB model as this oracle.
- For estimated dispersion, build a small independent profile reference that optimizes log phi externally and refits the explicit-dummy Statsmodels GLM at each candidate. Compare nuisance/donor coefficients, eta, means, dispersion, exact NLL, weights, score residuals, and convergence within prespecified tolerances.
- Verify `phi<=0`/nonfinite rejection, zero-total exclusion, one-cell/no-state-variation reporting, insufficient informative donors, weighted-rank failure, mean/dispersion nonconvergence, derivative/backtracking failure, and both positive dispersion boundaries all return non-success with no p-value surface.
- Verify eager/JIT equality and stable array-only kernel PyTree structure across success/failure, then host result/status/message mapping separately. Require finite gradients/JVPs for the fixed-dispersion likelihood and reduced solve across the small-phi branch, explicit predictor-overflow/underflow failure codes without `NaN` loop contamination, analytic expected cross-information zero, and nonzero observed mixed derivatives on a fixed sample.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cell_nb.py -x`
- Expected red: absorbed-NB fit/reference cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cell_nb.py`
- Expected green: all grouped, boundary, fixed-dispersion, alternating-fit, status, JIT, and derivative tests pass.

**Commit:** `feat: fit absorbed cell-level NB2 null`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_5 -->
### Task 5: Run Phase 3 regression and quality gates

**Verifies:** cis-state.AC2.1, cis-state.AC2.2, cis-state.AC2.3, cis-state.AC2.4, cis-state.AC2.5, cis-state.AC2.6

**Files:**
- Verify only; no planned file changes.

**Implementation:**
- Confirm the existing `GeneralizedLinearModel`, pseudobulk result types, and solver public contracts remain unchanged.
- Audit traced code for dynamic boolean slicing, Python branching on arrays, non-static `num_segments`, and failure-dependent PyTree structure.
- Review all modified application source classifications and keep `/Users/nicholas/Projects/jaxqtl/state.md` untracked/unstaged.

**Verification:**
- Run: `uv run pytest -p no:capture tests/test_cell_nb.py tests/test_optimize.py tests/test_lm_glm.py tests/test_families.py`
- Expected: absorbed-NB and adjacent numerical regressions pass.
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
- Verifies: cis-state.AC2.1, cis-state.AC2.2, cis-state.AC2.3, cis-state.AC2.4, cis-state.AC2.5, cis-state.AC2.6.

**Commit:** None unless verification finds a required fix; repeat the owning task's tests and commit the narrow correction.
<!-- END_TASK_5 -->
