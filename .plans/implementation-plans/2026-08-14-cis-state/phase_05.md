# Cis-by-state Phase 5 Implementation Plan

**Goal:** Calibrate the efficient omnibus statistic with reproducible donor multipliers and matrix-free donor-space moment approximations.

**Architecture:** Add one functional-core calibration module that consumes the Phase 4 private contribution operator and colocates deterministic blocking, trace accumulation, approximation formulas, configs, statuses, and results. Expose multiplier and moment entry points separately with explicit configuration; no calibration method becomes the default until Phase 7 simulation evidence.

**Tech Stack:** Python 3.11+, JAX 0.10+, Equinox, NumPy, SciPy reference tests, pytest, Ruff, ty

**Scope:** 7 phases from the original design; this file implements phase 5 of 7.

**Codebase verified:** 2026-08-14 16:32 PDT

---

## Review Profile

- Profile: `inference`
- Why this profile: This phase implements stochastic calibration, trace estimation, and moment-tail approximations whose numerical behavior and inferential interpretation both require specialist review.

## Acceptance Criteria Coverage

This phase implements and tests:

### cis-state.AC4: Calibration

- **cis-state.AC4.1 Success:** Multiplier calibration is reproducible from its seed, performs no null refits, and uses the specified corrected p-value.
- **cis-state.AC4.2 Edge:** Output reports bootstrap resolution and warns when an interpretation threshold is smaller.
- **cis-state.AC4.3 Scale:** Replicate and variant blocking does not change results or allocate `B x p x r`.
- **cis-state.AC4.4 Success:** Exact and estimated trace moments agree with dense donor-space references within numerical or Monte Carlo error.
- **cis-state.AC4.5 Failure:** Invalid three-moment parameters trigger a documented fallback or failure.

`cis-state.AC4.6` remains deferred: Phase 7 selects the default only from prespecified simulation evidence.

---

<!-- START_TASK_1 -->
### Task 1: Implement deterministic blocked donor-multiplier calibration

**Verifies:** cis-state.AC4.1, cis-state.AC4.2, cis-state.AC4.3

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/_cluster_calibration.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/_cluster_score.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/__init__.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_cluster_calibration.py` (unit/integration)

**Implementation:**
- Start `_cluster_calibration.py` with `# pattern: Functional Core`; preserve the Phase 4 classifications in `_cluster_score.py` and `hypothesis/__init__.py`.
- Keep multiplier and moment algorithms, immutable required-field configs/results, integer kernel status constants, host-visible status mapping, tail helpers, and trace accumulators in this one calibration module. Do not create bootstrap, moment, trace, status, or type-only leaf modules. Every jitted kernel returns array status codes plus fixed-shape diagnostics; the non-jitted public wrappers map codes to enums/actionable messages only after `device_get`.
- Export `calibrate_cis_state_multiplier(G_raw, S, donor_index, null_result, key, *, variant_block_size, config)` plus immutable `MultiplierCalibrationConfig`/result/status. Require config fields `num_replicates`, `replicate_block_size`, and optional `interpretation_threshold`; validate positive block/count and a finite threshold in `(0, 1)` when supplied.
- At both public calibration boundaries require x64 enabled and a float64 successful null result, promote accepted real `G_raw`/`S` arrays to float64, and launch only float64 score/operator/calibration kernels. Disabled x64, bool/complex inputs, or dtype-incompatible null results fail on the host before random draws.
- Call Phase 4 `_build_score_state` exactly once, retain its score result, and consume its private operator immediately. The calibration signature accepts a completed null result rather than raw outcomes/model objects, so it has no null-refit surface.
- Extend the private contribution operator only as needed with a bounded multiplier-block squared-adjoint action. For replicate signs `Xi` shaped `(replicate_block_size, N)`, accumulate `||Xi @ T_b||^2` over variant blocks and discard each `(replicate_block_size, variant_block_size*r)` temporary. Never allocate or return all `B x p x r` scores.
- Generate replicate `b` from `jax.random.fold_in(key, b)` and draw its `N` Rademacher signs; never split once per processing block. Host-pad the final replicate block with global indices and a validity mask so replicate/variant block sizes or scheduling cannot change the generated sequence.
- Count ties with `Q_b >= Q_observed` exactly and return `p = (1 + exceedances) / (B + 1)` after all requested replicates; do not early-stop. Report exact resolution `1 / (B + 1)`, approximate binomial Monte Carlo SE `sqrt(p * (1 - p) / B)`, and `resolution_warning = interpretation_threshold < resolution` when a threshold is supplied.
- Return a stable host-side non-success result with `pvalue=NaN` for upstream score/operator/nonfinite-bootstrap failures; structural config errors raise before tracing. Keep warnings distinct from fatal status and never place a runtime-selected Python enum/string in a transformed kernel result.

**Testing:**
- Write a tiny explicit-`T` multiplier oracle first and confirm behavior fails before implementation.
- Generate the same per-replicate signs by global folded index and compare every tiny `Q_b`, tie handling, integer exceedance count, corrected p-value, exact resolution, MC SE, and threshold warning.
- Verify the API consumes an already fitted null result and has no fit/refit callback or raw-y input. Assert that all score-state construction occurs once and calibration changes no null-result fields.
- With the same key and replicate count, compare replicate block sizes `1`, divisor, nondivisor, and `>= B`, plus variant block sizes `1`, divisor, nondivisor, and `>= p`; require bitwise-identical draws and numerically identical results. Verify a different key changes draws.
- Record allocations and forbid `B x p x r`, full `T`, `N x (p*r)`, and unbounded replicate-feature arrays; require the largest interaction temporary to be bounded by both configured blocks. Verify eager/JIT and padded-final-block behavior on the array-only kernel, then test host enum/message mapping separately.
- Verify isolated disabled-x64 failure, float32 public promotion, internal float32 launch rejection, and float64 dtype/backend reporting without changing folded random draws.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cluster_calibration.py -x`
- Expected red: multiplier/config/result cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cluster_calibration.py`
- Expected green: multiplier, reproducibility, blocking, resolution, and failure tests pass.

**Commit:** `feat: add donor multiplier calibration`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Estimate donor-space spectral trace moments

**Verifies:** cis-state.AC4.4

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/_cluster_calibration.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/_cluster_score.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cluster_calibration.py` (unit/integration)

**Implementation:**
- Export `calibrate_cis_state_moments(G_raw, S, donor_index, null_result, key, *, variant_block_size, config)` plus immutable `MomentCalibrationConfig`/result. Require explicit fields `num_probes >= 2`, positive `probe_block_size`, approximation (`two-moment` or `three-moment`), invalid-three-moment policy (`fallback-two-moment` or `fail`), and positive finite `validity_rtol`.
- Reuse Phase 4 `exact_trace()` for `c1 = trace(B) = ||T||_F^2` with zero estimator SE, where `B = T T.T`. Do not form `T` or `B`.
- For independent Rademacher probe `z_s`, estimate `c2 = trace(B^2)` with `h2_s = ||B z_s||^2`. When three moments are requested, reuse `B z_s` and one additional Gram action to estimate `c3 = trace(B^3)` with `h3_s = (B z_s).T B(B z_s)`.
- Generate probe `s` with `jax.random.fold_in(key, s)` so probe blocking does not change samples. Hold at most `probe_block_size x N` probe/Gram arrays plus the operator's bounded probe/variant temporary; never allocate `N x N`.
- Merge count/mean/M2 across padded probe blocks using a numerically stable Welford accumulator. Report empirical independent-probe Monte Carlo SE `sqrt(sample_variance(h, ddof=1) / R)` and relative SE for each estimated trace. Clamp only negative M2 within documented roundoff tolerance and record that adjustment; fail on material negative or nonfinite probes/accumulators.
- Document that reusing the same probe for `h2` and `h3` preserves marginal unbiasedness but correlates their errors. Report per-trace Monte Carlo diagnostics only; do not label them analytic concentration bounds or propagated uncertainty for fitted approximation parameters/p-values.

**Testing:**
- Add dense donor-space trace tests first and observe failure before implementation.
- On a tiny explicit `T`, compare exact `c1` and dense eigenvalue `c1/c2/c3 = sum(lambda^1/2/3)`. For a sufficiently large fixed probe count, require estimated traces to agree within a prespecified multiple of their reported MC SE.
- Separately feed stored NumPy probe values through the streaming accumulator and compare means, `ddof=1` variances, absolute SEs, and relative SEs across probe block sizes.
- Verify folded-index determinism/block invariance, two-moment avoidance of unnecessary second Gram actions, same-probe c2/c3 reuse, roundoff adjustment diagnostics, nonfinite/material-negative failures, eager/JIT equality, and the absence of `N x N`/full-`T` allocations.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cluster_calibration.py -x`
- Expected red: moment/trace tests fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cluster_calibration.py`
- Expected green: multiplier and trace-estimation tests pass.

**Commit:** `feat: estimate donor score trace moments`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Add two- and three-moment chi-square calibration

**Verifies:** cis-state.AC4.4, cis-state.AC4.5

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/_cluster_calibration.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/__init__.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cluster_calibration.py` (unit)

**Implementation:**
- Name `c_l` explicitly as spectral trace powers `trace(B^l)`, not cumulants. For `Q = sum lambda_i * chi2_1`, document cumulants `kappa_l = 2^(l-1) * (l-1)! * c_l` to prevent factor-of-two/four mistakes.
- Implement two-moment Satterthwaite parameters with ratio algebra: `a2=c2/c1` and `nu2=c1/a2`, never the overflow-prone direct power `c1^2/c2`. Evaluate the upper tail with `jax.scipy.stats.chi2.sf(Q/a2, df=nu2)` rather than `1-cdf`, after scale-normalizing the statistic. Require positive finite moments/parameters and a finite p-value in `[0, 1]`.
- Implement the literal central shifted/scaled three-trace-moment match, not Liu-Tang-Zhang's four-cumulant noncentral approximation. Compute `a3=c3/c2`, `log_nu3=log(c2)-2*log(a3)`, `nu3=exp(log_nu3)` only after checking it against the float64 log-domain, and `delta=c1-c2/a3`; never form `c2^3`, `c3^2`, or `c2^2`. Use p-value `1` for `Q <= delta`, otherwise evaluate the normalized tail `chi2.sf((Q-delta)/a3, df=nu3)`.
- Require positive finite `c1/c2/c3/a3/nu3` and PSD inequality `delta >= -validity_rtol * max(1, c1)`. Snap only an in-tolerance negative shift to zero and expose a boundary-adjustment diagnostic. Never clip moments, scale, degrees of freedom, shift, or p-value silently.
- On invalid three-moment parameters, obey `invalid_three_moment`: either run the already validated two-moment branch and report `requested=three-moment`, `used=two-moment`, fallback status/reason/warning, or return failure with `pvalue=NaN`. Invalid two-moment parameters always fail.
- Return nested score result, statistic, p-value, requested/used approximation, trace values/SEs/relative SEs, probes, scale/df/shift, fallback/roundoff flags, warnings, status, and actionable host message from the host wrapper. The numerical approximation kernel returns only arrays/codes with invariant PyTree structure. Do not choose a method default in this phase.

**Testing:**
- Add approximation/validity tests first and confirm failure.
- Compare two-moment parameters/tails with `scipy.stats.chi2.sf` on exact dense spectra. Compare constructed valid three-moment parameters and tails with SciPy's central shifted/scaled chi-square for statistics below, at, and above the shift.
- Verify the trace-power-to-cumulant relation, exact PSD equality/boundary cases, materially invalid estimated inequality, nonpositive/nonfinite moments/parameters, in-tolerance shift adjustment, both explicit fallback/fail policies, and nonfinite/out-of-range tail failures. Add common spectral rescalings spanning at least `1e-100` through `1e100` wherever `c1/c2/c3` remain representable; require scale-invariant degrees of freedom/tails and no overflow from avoidable intermediate powers.
- Assert the public exports present two explicit calibration functions/configs and no generic/default calibration selector.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cluster_calibration.py -x`
- Expected red: approximation/fallback cases fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cluster_calibration.py`
- Expected green: all multiplier, trace, approximation, and failure tests pass.

**Commit:** `feat: add donor moment calibration`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Run Phase 5 regression and quality gates

**Verifies:** cis-state.AC4.1, cis-state.AC4.2, cis-state.AC4.3, cis-state.AC4.4, cis-state.AC4.5

**Files:**
- Verify only; no planned file changes.

**Implementation:**
- Confirm calibration methods remain explicit and no runtime/speed-based default appears in public APIs or configs.
- Audit RNG generation for global folded indices, traced code for dynamic compaction, and allocation logs for prohibited full bootstrap/contribution/covariance arrays.
- Review source classifications and keep `/Users/nicholas/Projects/jaxqtl/state.md` untracked/unstaged.

**Verification:**
- Run: `uv run pytest -p no:capture tests/test_cluster_calibration.py tests/test_cluster_score.py tests/test_cell_nb.py`
- Expected: calibration, score, and null-fit tests pass.
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
- Verifies: cis-state.AC4.1, cis-state.AC4.2, cis-state.AC4.3, cis-state.AC4.4, cis-state.AC4.5.

**Commit:** None unless verification finds a required fix; repeat the owning task's tests and make a narrow correction commit.
<!-- END_TASK_4 -->
