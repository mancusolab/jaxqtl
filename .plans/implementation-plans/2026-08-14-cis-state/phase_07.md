# Cis-by-state Phase 7 Implementation Plan

**Goal:** Generate prespecified statistical evidence for calibration selection, demonstrate the OneK1K shape/memory contract, publish the chosen defaults, and document the supported interpretation.

**Architecture:** Extend the existing pure simulation module with the cis-state generative contract. Keep prespecified statistical decision rules in a small functional script module and expensive orchestration/publication in a separate shell. Add one subprocess-based benchmark shell for isolated peak-memory and cold/warm timing. Select defaults only from a versioned evidence report, then update public adapters and documentation.

**Tech Stack:** Python 3.11+, JAX 0.10+, Equinox, NumPy, SciPy, Polars, PyArrow, SciPy sparse, pytest, Ruff, ty, Zensical

**Scope:** 7 phases from the original design; this file implements phase 7 of 7.

**Codebase verified:** 2026-08-14 18:16 PDT

---

## Review Profile

- Profile: `full`
- Why this profile: This phase adds a public simulation contract, prespecified inferential evidence gates, resumable publication, scale benchmarks, evidence-selected defaults, and documentation.

## Acceptance Criteria Coverage

This phase implements and tests:

### cis-state.AC4: Calibration

- **cis-state.AC4.6 Decision:** The default calibration is selected from the prespecified simulation evidence, not runtime.

### cis-state.AC6: Scale and engineering

- **cis-state.AC6.1 Scale:** OneK1K-style benchmarks complete without prohibited allocation shapes.
- **cis-state.AC6.2 Observability:** Benchmarks report state construction, compilation, warm runtime, peak memory, and observed allocation shapes.
- **cis-state.AC6.3 Scale:** Fixed block shapes and buckets prevent per-gene recompilation growth.
- **cis-state.AC6.4 Quality:** Tests, Ruff checks, formatting validation, and `ty` pass.
- **cis-state.AC6.5 Documentation:** Documentation states the estimand, assumptions, calibration resolution, fixed-state interpretation, and kinship limitation.

### cis-state.AC7: Simulation validation

- **cis-state.AC7.1 Reproducibility:** Simulation accepts an explicit JAX key and reproduces every generated dataset.
- **cis-state.AC7.2 Coverage:** It covers the NB null, overdispersion, unequal depths and cell counts, within-donor dependence, population/state confounding, zero-count donors, and increasing `tau`.
- **cis-state.AC7.3 Evidence:** Reports include rejection rates, Monte Carlo intervals, power, convergence failures, and effective dimensions.
- **cis-state.AC7.4 Comparison:** Moment calibration is compared directly with donor multiplier calibration.
- **cis-state.AC7.5 Decision:** The selected default controls type-I error within prespecified binomial Monte Carlo tolerances.

---

<!-- START_TASK_1 -->
### Task 1: Add the reproducible cis-state generative simulation contract

**Verifies:** cis-state.AC7.1, cis-state.AC7.2

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/sim.py`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_sim_cis_state.py` (unit/statistical)
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_sim.py` (regression)

**Implementation:**
- Add `# pattern: Functional Core` to `sim.py`; preserve `simulate_pheno` behavior. Keep the new generative model in this existing module unless the completed implementation demonstrates a repeated simulation subdomain that makes it materially incohesive; do not preemptively create a `sim/` package.
- Add immutable `CisStateSimulationParams`, `CisStateSimulationConfig`, scenario metadata, and `CisStateSimulation`, and export `simulate_cis_state(key, params, config) -> CisStateSimulation`.
- Give every exported simulation object a raw public docstring compatible with the repository's mkdocstrings style. Document arguments/fields, returned shapes/dtypes, validation failures, NB2 parameterization, exact interaction coordinate, named-key determinism/replay, supported backend/dtype, and the distinction between simulation truth and fitted-analysis inputs.
- Validate positive donor/cell/variant/state dimensions, MAF/LD ranges, required donor-covariate dimension `>=1`, positive library-size parameters, strictly positive finite NB2 dispersion, and compatible effect shapes before sampling. Use a small positive dispersion for near-Poisson scenarios; never pass `phi=0` to the NB sampler.
- Before coercion or a stochastic operation, reject bool/complex numerical parameter and effect leaves; accept integers only for documented discrete fields and promote accepted real inexact inputs to float64. Require `jax.config.x64_enabled`, place an actual scalar float64 probe on the resolved current device, and verify both its dtype and CPU/GPU backend; fail at the public host boundary on any mismatch rather than accepting truncated draws. Construct every real-valued parameter, random variate, predictor component, `G_std`, `S`, offset, mean, dispersion, and coefficient array as float64; counts/dosages/indices retain their documented integer dtypes. Record the probed x64, backend/device, and real dtype in scenario metadata.
- Generate donor-level dosage with explicit MAF and LD parameters; donor ancestry/covariates; unequal donor cell counts; dense zero-based `donor_index`; library-size offsets; state factors with configurable ancestry-correlated donor distributions; optional cell covariates; unrestricted donor intercepts; donor-covariate-by-state nuisance terms; and explicit genotype-by-state `Gamma`.
- Define `interaction_variance=tau` unambiguously: sample each alternative coefficient as `sqrt(tau) * Normal(0,1)`; the null uses an exactly zero matrix. Standardize raw dosage over the complete aligned `N`-donor cohort with the Phase 4 population-RMS (`ddof=0`) rule, and define the generative interaction as `eta_interaction[c] = sum_jk G_std[donor_index[c],j] * Gamma[j,k] * S[c,k]`, using natural unwhitened state factors. The whole-kernel `c_G/c_S` scaling remains score normalization only and does not enter the generating predictor. Store raw dosage, `G_std`, genotype means/scales, natural `S`, `Gamma`, and the interaction component so the coordinate convention is reconstructible. Retain every additive linear-predictor component so omitted-term/confounding tests can distinguish truth from analysis inputs.
- Draw counts with the repository's NB2 parameterization `Var(y|mu)=mu+phi*mu^2` through `NegativeBinomial.sample`. Add controlled misspecification through donor/substate latent slopes or correlated substate effects, and an explicit edge control that can yield/force recorded zero-total donors without mislabeling it as the fitted NB model.
- Derive subkeys by stable named fold-in indices rather than order-sensitive sequential loops; metadata records root key material, scenario ID, replicate ID, parameters, and replay information. Return `y`, `G`, `S`, `donor_index`, `offset`, cell covariates, required donor covariates, `mu`, `phi`, `Gamma`, predictor components, informative-donor truth, and dimensions.

**Testing:**
- Write deterministic contract tests first and observe failures before implementation.
- Compare every returned shape and recomposed linear predictor/mean with stored components. Assert exact `Gamma=0` under the null and `Gamma/sqrt(tau)` sampling semantics under alternatives.
- Verify bitwise replay for the same key/config, changed draws for changed scenario/replicate fold-ins, and invariance of one component's subkey to enabling unrelated optional components.
- Test NB2 empirical mean/variance on a reduced repeated-draw fixture; genotype MAF/LD moments; cell-count/depth heterogeneity; ancestry/state/genotype correlation; latent within-donor dependence; and exact truth/accounting for zero-total, singleton, and no-state-variation donors.
- Cover invalid dimensions/domains/nonfinite parameters and ensure the established `simulate_pheno` tests remain unchanged.
- In isolated subprocesses, disable/unavailable x64 or force a float64-probe/backend mismatch and require failure before the first draw. Reject boolean/complex scalar and effect-array leaves before JAX coercion. With x64 enabled, assert every returned real leaf is float64 on CPU and any available x64 GPU, replay remains bitwise stable on the same backend, and metadata records the verified setup. Keep global JAX configuration changes isolated from the suite.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_sim_cis_state.py -x`
- Expected red: new simulation contract and scenario assertions fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_sim_cis_state.py tests/test_sim.py`
- Expected green: new and established simulation tests pass.

**Commit:** `feat: simulate cis-state interactions`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Prespecify calibration experiments and decision rules

**Verifies:** cis-state.AC4.6, cis-state.AC7.2, cis-state.AC7.3, cis-state.AC7.4, cis-state.AC7.5

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/scripts/cis_state_validation_core.py`
- Create: `/Users/nicholas/Projects/jaxqtl/.scientific/validation/cis-state/calibration-config-v1.json`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_scripts/test_cis_state_validation_core.py` (unit)

**Implementation:**
- Start `cis_state_validation_core.py` with `# pattern: Functional Core`. Colocate immutable closed-schema validation-config/report records, deterministic scenario expansion, online rejection/power/convergence/recovery summaries, exact-binomial interval formatting, paired-method comparisons, trace-error summaries, and the calibration decision gate. Keep random simulation/execution and file writes out of this module.
- Commit the version-1 scientific prespecification before generating results. Give it `generation=1`, null `supersedes_report_path`, `supersedes_report_sha256`, and `supersession_reason` fields, and a required `source_inventory_paths=null` materialization sentinel; reject every missing or unknown field. Task 2 tests canonical JSON encoding, but no run or report may treat the config as final until Task 3 replaces only that sentinel with the exact source inventory and computes the final config SHA-256. Pin primary null levels `(0.05, 0.01)`, `2,000` independent simulation replicates per primary scenario, and tolerance multiplier `3`. These are the only evidentially validated significance levels; the report must not relabel their convex hull as a continuously validated range. Never alter these values in response to observed results.
- Pin all generative and analysis settings in the JSON, with no hidden defaults: root seed `20260814`; CPU/x64/float64; `N=120`, `p=64`, `r=8`, three required donor covariates, two cell covariates, and the exact per-scenario cell-count/depth/MAF/LD/effect/dispersion parameters; `QRSolve`; `min_informative_donors=30`; rank rule `max(M_active,d)*eps(float64)`; NB mean/dispersion iteration caps `100/50`, predictor/coefficient relative tolerances `1e-7`, `delta_log_phi` tolerance `1e-6`, positive phi bounds `[1e-8,1e3]`, at most `20` backtracking steps, scaled backward-error tolerance `1e-8`, and Cholesky accuracy budget `1e-8`; variant block/capacity `64/64`; donor multiplier `B=1,999`, replicate block `64`, interpretation threshold `0.01`; and moment probes/block `512/64`, requested `three-moment`, invalid policy `fallback-two-moment`, validity tolerance `1e-10`. Pin schema/package versions and ensure any later config-field rename preserves these values and canonical meaning.
- Pin selectors for one canonical numerical-source inventory broad enough to cover transitive algorithm dependencies for both calibration and performance evidence: `src/jaxqtl/sim.py`; every tracked `*.py` recursively under `src/jaxqtl/state/`, `src/jaxqtl/distribution/`, `src/jaxqtl/infer/`, and `src/jaxqtl/hypothesis/`; plus `scripts/cis_state_validation_core.py`, `scripts/validate_cis_state.py`, and this calibration config. The only exclusions inside those roots are the later evidence-binding shell files `src/jaxqtl/hypothesis/_cis_state_defaults.py` and `src/jaxqtl/hypothesis/__init__.py`; list those exact paths in the schema. This deliberately includes reused implementations such as `infer/_solve.py`, distribution links/families, and state PFlog/operator/factorization even when a particular evidence profile does not execute every member. Task 2 freezes the selectors and exclusions, not an impossible pre-Task-3 file list. Task 3 materializes the exact sorted paths from its complete candidate tree, commits the finalized config with the harness, and verifies the clean committed tree expands to the identical list. Task 4 hashes each normalized path length/path plus byte length/raw bytes from that clean commit. Reject a missing, renamed, untracked replacement, dirty member, or unexpected source-root file other than the two declared exclusions before a full run; generated evidence, documentation, CLI/I/O shells, tests, and benchmark-only harness/config files are outside the enumerated roots.
- Pin the reference generator completely. Every reference replicate has exactly `32` cells per donor; `log(library_size_c) = log(1000) + 0.3*z_c`, `z_c ~ Normal(0,1)`; and donor intercepts `alpha_i ~ Normal(-4.6, 0.4^2)`. The three ordered donor covariates are ancestry `a_i ~ Normal(0,1)`, an independent standardized-age variate `d2_i ~ Normal(0,1)`, and encoded sex `d3_i` sampled Rademacher `{-1,+1}`. The two ordered cell covariates are `c1_c ~ Normal(0,1)` and Rademacher `c2_c`. Do not center or rescale these generated analysis inputs after sampling.
- Pin natural state and genotype generation. Draw `epsilon_ck ~ Normal(0,1)` and set `S_ck = epsilon_ck + lambda_S*a_i*v_k` for the donor owning cell `c`, with `v=(1,-1,0,0,0,0,0,0)/sqrt(2)`; do not whiten or column-standardize `S`. Draw variant MAFs independently as `m_j ~ Uniform(0.1,0.4)`. For every donor, generate a stationary Gaussian AR(1) sequence `z_i1 ~ Normal(0,1)`, `z_ij=rho*z_i,j-1+sqrt(1-rho^2)*epsilon_ij`, form `g_latent_ij=(z_ij+lambda_G*a_i)/sqrt(1+lambda_G^2)`, and convert `Phi(g_latent_ij)` to HWE dosages `0/1/2` at cumulative thresholds `(1-m_j)^2` and `1-m_j^2`. This is the only genotype/ancestry/LD construction used by the evidence run.
- Pin every noninteraction coefficient in C-order. Use state main effects `beta_S=(0.20,-0.15,0.10,-0.08,0.06,-0.04,0.03,-0.02)`, cell-covariate effects `beta_C=(0.12,-0.08)`, and donor-covariate-by-state matrix rows `theta_1=(0.10,-0.08,0.06,-0.04,0.03,-0.02,0.01,-0.01)`, `theta_2=(-0.06,0.05,-0.04,0.03,-0.02,0.015,-0.01,0.005)`, and `theta_3=(0.04,0.03,-0.03,-0.02,0.02,0.01,-0.01,-0.005)`. The latent-dependence term is an omitted donor-specific slope `b_i*S_c1`, with `b_i ~ Normal(0,sigma_latent^2)`; no other nuisance or latent effect may be supplied by an implementation default.
- Pin primary null scenarios: fitted NB reference; strong overdispersion; unequal depth; unequal donor cell counts; added latent within-donor state dependence; population/state confounding with the required adjustment; and a combined stress case. Use reference values `phi=0.2`, `sigma_latent=0`, `lambda_G=lambda_S=0`, and the fixed cells/depth above. Change only the named axis for single-stress scenarios: `phi=2.0`; library log-SD `1.0`; or cell counts `clip(round(exp(log(32)+z_i)),8,96)` with `z_i ~ Normal(0,1)`; or `sigma_latent=0.5`; or `lambda_G=lambda_S=0.8`. The combined case uses `phi=1.0`, library log-SD `1.0`, that exact unequal-cell rule, `sigma_latent=0.5`, and both loadings `0.8`. All other distributions and coefficient vectors remain the pinned reference values.
- Pin exactly `128` replicates for each edge scenario and exact deterministic overrides after reference generation: `zero-total` forces all outcomes for donor indices `0:24` to zero; `singleton` assigns one cell to donor indices `0:24` and `32` to every other donor; `no-state-variation` replaces every state row for donor indices `0:24` by that donor's ancestry-only row `lambda_S*a_i*v`; `invalid-genotype` makes variant `0` constant one, sets `G[0,1]=NaN`, and sets `G[1,2]=+Inf`, leaving exactly `61` expected valid variants; `insufficient-informative` keeps donors `0:29`, forces donors `29:120` to zero-total, and sets the first outcome of any retained donor with a sampled zero total to one; and `rank-boundary` sets state column `7` exactly equal to column `6`. The first four cases must produce the exact configured masks/counts and a valid result under the remaining informative cohort; insufficient-informative and rank-boundary must return their named failure codes and no p-value.
- Pin two deterministic dispersion-boundary fixtures, also at `128` replicates each. The lower fixture uses the production analysis bounds `[1e-8,1e3]`, sets every cell offset and true mean-effect coefficient to zero, and sets every outcome to `10`; require `dispersion_lower_bound`. The upper fixture is explicitly a controller-status validation case with bounds `[1e-8,10]`, not evidence that the production upper bound is routinely reached: set offsets/effects to zero and alternate outcomes `0,1_000_000` within every 32-cell donor. Construct its within-donor nuisance rows by drawing a fixed keyed `32 x 10` matrix, projecting it off both the constant vector and the zero/high outcome indicator, taking a reduced QR with largest-magnitude-entry-positive column signs, and assigning the first eight columns to `S` and last two to cell covariates; use the fixed donor covariates from the reference generator after requiring their realized matrix rank to be three. This makes the nuisance optimum zero and preserves the donor-only profile. Define the independent oracle as the float64 root of the analytic donor-profile score in `r=1/phi`, solved by bracketed Brent with `xtol=rtol=1e-14`, then evaluated with the stable NB2 likelihood: `phi=17.68700728544642` and per-donor NLL `284.78281765072495`. Require the unconstrained oracle to match with `log(phi)` `atol=1e-6, rtol=1e-6` and NLL `atol=1e-6, rtol=1e-8`, then require the constrained fit to return `dispersion_upper_bound` at `10`. Both boundary fixtures return no p-value.
- Pin a separate deterministic production-bound interior fixture at `128` replicates. Use the same zero offsets/effects, donor covariates, and projection/QR construction, but project off a one-high-cell indicator and set the first 31 outcomes within every donor to zero and the last to `1_000_000`. Using the same analytic-score/Brent/stable-likelihood oracle gives `phi=516.7234972369305` and per-donor NLL `21.124944061280054`; require a successful fit under `[1e-8,1e3]`, `log(phi)` agreement with `atol=1e-6, rtol=1e-6`, and total NLL agreement after multiplying the per-donor oracle by `120` with `atol=1e-5, rtol=1e-8`. Store both oracle constructions, methods, values, and tolerances in the closed-schema config; do not infer production-bound behavior from the validation-only case.
- Pin a distinct paired power scenario at interaction variances `(0.0, 0.01, 0.04, 0.09)` with exactly `1,000` scheduled replicate indices `0:1000` at every point, including `tau=0`. This power-null arm is generated under scenario ID `power-reference` and is not borrowed from either the first 1,000 or all 2,000 replicates of any primary-null scenario. Within replicate index `b`, reuse the same generated `G_raw`, `G_std`, `S`, covariates, offsets, latent effects, and standard-normal coefficient direction `Z_Gamma` across all four points; set `Gamma_tau=sqrt(tau)*Z_Gamma`, and derive the count key by folding in the fixed tau-grid index. Thus every adjacent comparison has the same 1,000 paired replicate IDs and every marginal power/null denominator is 1,000. Report power and exact 95% binomial intervals. Require the highest-variance lower interval to exceed this paired `tau=0` upper interval and require every adjacent power change to be nonnegative within three standard errors of the sample mean of paired rejection-indicator differences. A material reversal fails power validation; record it without weakening or post-hoc changing the grid.
- Derive disjoint recorded top-level PRNG streams from every `(scenario_id, replicate_id)` key with fixed fold-in tags `data=0`, `multiplier=1`, `moment=2`, and `observation_diagnostic=3`. The simulator derives its own named component subkeys only below `data`; calibration code may never reuse a data or another calibration stream. Both production methods analyze the identical generated dataset for a replicate.
- Treat only `{0.01,0.05}` as evidentially validated significance levels; do not claim continuous validation over the interval. For each null method/level/scenario define the preregistered calibration band `abs(rejection-alpha) <= max(3*sqrt(alpha*(1-alpha)/R), 1/R)`. The rejection numerator counts scheduled replicates with a finite valid method p-value `<=alpha`; the denominator is all `R` scheduled replicates, so failures are never dropped. Also report success-conditioned rejection only as a diagnostic, SciPy exact 95% binomial intervals, and QQ/empirical-CDF summaries; the fixed unconditional band, not the interval or conditional rate, is the decision criterion.
- Track inference failure, multiplier-specific failure, moment-specific failure, and three-moment fallback separately over all scheduled replicates. A method is eligible only when its total unusable-p-value rate (common inference failure or its method-specific failure) is at most `1%`; moment additionally requires fallback at most `1%`. Define the paired moment-vs-multiplier difference from the all-scheduled-replicate rejection indicators and its sample SE. Moment is eligible only if both methods pass every primary null band, its failure/fallback gates pass, the absolute paired difference is within `max(3*SE_difference, 1/R)` at each primary scenario/level, and dense-subset `c2/c3` truth differs from estimates by no more than `4` reported Monte Carlo SE plus numerical tolerance. If moment is ineligible but multiplier passes its own null and failure gates, select multiplier. If multiplier fails either gate, select no default and fail the phase for diagnosis.
- For SIM-3, analyze each population/state-confounded dataset twice: the adjusted fit uses ordered donor covariates `(ancestry, standardized_age, encoded_sex)`, while the intentionally misspecified fit omits ancestry and retains `(standardized_age, encoded_sex)`. Use identical generated data and calibration keys for the pair. At level `0.05`, require the adjusted cluster method to pass its null band, the unadjusted rejection rate to exceed that band's upper endpoint, and the paired `unadjusted-adjusted` rejection difference to exceed `max(3*SE_difference,1/R)`; report the same paired contrast at `0.01` without adding a second pass criterion. This diagnostic is required evidence but cannot make an otherwise ineligible production method eligible.
- Add a simulation-only observation-level Rademacher diagnostic that multiplies cell-level efficient contributions independently with the same `B=1,999` and block size `64`. It is never exported as a production calibration or eligible default. SIM-2 must report that donor-cluster methods satisfy their prespecified null bands while this intentionally misspecified comparator shows material inflation above the upper band in at least the combined within-donor-dependence scenario.
- Define that diagnostic precisely with conceptual cell rows `e_c = q_c h0_c - (H0.T W X0) J_XX^{-1}(q_c x0_c)`, using the same normalized, donor-centered interaction features and nuisance information as Phase 4. Generate one Rademacher sign per cell and compute `||sum_c sign_c e_c||^2` in fixed replicate/variant blocks; never materialize `M x (p*r)`. Verify its observed contribution sum matches the efficient score before using it only as the deliberately independence-misspecified SIM-2 comparator.
- Prespecify SIM-7 without assuming finite-sample MLE unbiasedness. Transform truth to the exact analysis coordinate and report signed bias, RMSE, and empirical quantiles for six separately pooled classes: the `8` state coefficients, `2` cell-covariate coefficients, `24` donor-covariate-by-state coefficients in C-order, `120` donor intercepts, `log(phi)`, and per-cell relative mean prediction. For the last class define signed bias `(mu_hat-mu_true)/max(mu_true,1e-12)` and RMSE from the same cellwise relative error. Pool coordinates only within the named class after first computing one class metric per replicate; also report per-coordinate summaries without using them as separate pass/fail tests.
- Add paired recovery-only reference datasets with exactly `32`, `64`, and `128` cells per donor for replicate indices `0:1000`, reusing each replicate's donor/genotype/coefficient truth and named random streams across sizes. Record success/failure separately at every size and `any_triplet_failure` over all `1,000` scheduled replicate triplets. The SIM-7 recovery gate requires zero failed triplets: never drop, impute, or condition away a failed 32/64/128 fit. Only after that gate passes, compute all paired summaries with denominator `1,000`. For each of the six classes, gate consistency with `mean(RMSE_128 - 0.9*RMSE_32) <= 3*SE_of_paired_difference + 1e-6`; the six one-sided three-SE class gates have union-bound error below `1%`. The 64-cell result is reported as an intermediate trend but is not an extra multiplicity-bearing gate. The zero-failure recovery-triplet gate is intentionally stricter than the separate at-most-`1%` production-scenario numerical-failure diagnostic. This class-level consistency criterion and dense-oracle equivalence are the other recovery pass conditions; finite-sample truth bias is descriptive and may not block a correct estimator by itself.
- On 32-cell reference replicate indices `0:128`, compare the optimized fit with the explicit dense donor-dummy/profile oracle using simultaneous maximum-over-field numerical tolerances: nuisance coefficients and donor intercepts `atol=1e-6, rtol=1e-5`; `log(phi)` `atol=1e-6, rtol=1e-5`; `eta` and `mu` `atol=1e-7, rtol=1e-6`; and total NLL `atol=1e-6, rtol=1e-8`. These are deterministic implementation-equivalence checks, not zero-bias tests. On 32-cell reference indices `0:64`, compare exact dense `c2/c3` with trace estimates. Run the observation-level SIM-2 comparator on all `2,000` combined-stress replicates. All index ranges, tolerances, counts, class definitions, and the `0.9` consistency factor are closed-schema config fields. A material consistency, dense-oracle, or numerical-failure gate blocks the full validation result.
- Report recovery errors/bands, effective donors/variants/states, trace SEs, fallback reasons, convergence/status frequencies, observation-level diagnostics, and replay keys. Never claim trace SEs are p-value uncertainty or analytic concentration bounds.

**Testing:**
- Add pure decision-rule tests first and observe failure before implementation.
- Test rejection of missing/unknown config fields, canonical prespecification encoding, the source selector/exclusion schema and required null materialization sentinel, rejection of a sentinel config as final evidence, every pinned distribution/vector/edge override/subset/tolerance, the independent upper-fixture and production-interior profile optima/NLLs, exact interaction-coordinate reconstruction, disjoint named subkeys, stable scenario expansion and replay IDs, the distinct `power-reference` `tau=0` arm and all four 1,000-member paired ID sets, online aggregation equivalence to stored NumPy arrays, exact 95% binomial intervals, three-SE bands, paired-difference SE, QQ summaries, power interval criterion, edge-status counts, adjusted-versus-unadjusted SIM-3, observation-level inflation, zero-failure/all-`1000`-denominator recovery triplets, class-level paired recovery consistency and multiplicity handling, dense-oracle maximum-field tolerances, and trace-within-MC-error checks. Test the pure digest function in Task 2 only with synthetic explicit `(path, bytes)` records; actual-tree expansion and per-final-member mutation coverage belong exclusively to Task 3. A synthetic single failed triplet must block SIM-7 rather than disappear from aggregation.
- Construct synthetic reports for moment selected, multiplier fallback, multiplier failure/no-default, nominal success-conditioned rejection with excessive failures, boundary equality, excessive fallback/numeric failure, invalid/incomplete scenarios, changed config hash, and attempted post-hoc parameter changes.
- Assert the gate never reads timing fields and cannot select a method from speed.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_scripts/test_cis_state_validation_core.py -x`
- Expected red: preregistration, aggregation, and decision-gate assertions fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_scripts/test_cis_state_validation_core.py`
- Expected green: all pure evidence and decision tests pass.

**Commit:** `test: prespecify cis-state calibration validation`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Implement resumable calibration-validation orchestration

**Verifies:** cis-state.AC4.6, cis-state.AC7.1, cis-state.AC7.2, cis-state.AC7.3, cis-state.AC7.4, cis-state.AC7.5

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/scripts/validate_cis_state.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/.scientific/validation/cis-state/calibration-config-v1.json`
- Modify: `/Users/nicholas/Projects/jaxqtl/.gitignore`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_scripts/test_validate_cis_state.py` (integration/system)

**Implementation:**
- Start `validate_cis_state.py` with `# pattern: Imperative Shell` and a standard-library-only bootstrap layer: module top level imports no JAX/jaxqtl code. Parse/validate the config or binding with `argparse`/`json`, set the pinned JAX platform and x64 environment before lazy-importing JAX/jaxqtl workers, then require `jax.config.x64_enabled`, the exact configured backend, and an actual JAX float64 probe. Only after that probe may the worker expand scenarios, derive each replicate key from `fold_in(root_key, scenario_index)` then `fold_in(..., replicate_index)`, derive the fixed named data/multiplier/moment/observation streams, and execute both calibrations/recovery diagnostics.
- Materialize `source_inventory_paths` as Task 3's only permitted scientific-config edit: expand the Task 2 selectors/exclusions against the complete staged Task 3 tree, sort the paths, replace the null sentinel, and assert every other canonical config field is byte-for-byte unchanged from the Task 2 commit. Commit the harness and finalized config together, then verify expansion from the clean commit produces exactly that list and compute the final canonical config SHA-256. The full/reduced runners reject the null sentinel, path drift, or a config whose scientific fields changed during materialization.
- Provide `--profile reduced|full`, `--config`, `--out`, optional root-seed/platform equality assertions, and bounded replicate-batch options. For a full run, and by default for reduced runs, the effective root seed/platform/x64/dtype come only from the immutable config; any supplied seed/platform value must equal it and can never override it. Reduced mode changes only its closed-schema replicate counts, uses the same setup/scenarios/logic, is labeled non-evidentiary, and may never select/update a default.
- Require an explicit `--checkpoint-dir` that resolves to a sibling of `--out` and is neither equal to it nor an ancestor/descendant of it. The final `--out` path remains nonexistent throughout simulation and checkpoint reuse. Store resumable immutable partitions in that distinct checkpoint directory by `(scenario_id, replicate_start, replicate_stop)`. Each partition is written through its own staging path, contains fixed-schema scalar records/replay keys only, and embeds config, statistical-core, environment, schema, range, row-count, and payload hashes. Before reuse, validate all hashes, nonoverlapping exact ranges, and code/config compatibility; never append to or mutate a completed partition.
- Add only `.scientific/validation/cis-state/.calibration-run-v*.checkpoints/` to `.gitignore`. Evidence configs, final `calibration-run-v*/`, benchmark configs/reports, and the selected-evidence binding remain trackable. This narrow rule lets an interrupted validated checkpoint set coexist with Task 4's clean tracked/source preflight; checkpoint reuse still depends on the shell's cryptographic/range validation, never on Git ignore status.
- Accept `--binding` as mutually exclusive with `--config` from the start. It reads the Task 7 canonical selected-evidence JSON and resolves the explicitly named calibration config generation after validating the binding/config/report hashes; it never scans filenames or mtimes. Tasks 3-4 use `--config` before a binding exists, while the final audit uses `--binding`.
- Atomically assemble a nonexistent final evidence directory only after validated checkpoint partitions cover every required range exactly once. Copy the validated partitions and combined Parquet into the final sibling staging directory, write the JSON summary/decision manifest last, validate hashes/counts/completeness, and publish. On success remove the checkpoint directory only after the final artifact is visible; a failed/interrupted run retains reusable validated checkpoints but cannot emit a decision-bearing final artifact.
- Record requested and verified x64/platform/device/real dtype plus package versions, config hash, source commit, canonical statistical-core tree digest, requested/completed replicates, all rejection/power/convergence/recovery/effective-dimension/trace/fallback/observation-level summaries, exact intervals, paired comparisons, and the decision with reasons. Every checkpoint embeds the same verified setup and is incompatible with a setup mismatch.
- Do not suppress failed replicates: record their replay key, inference status, separate multiplier/moment statuses, dimensions, and message; include them in the all-scheduled denominator and the prespecified per-method failure gates. No scenario/config replacement is allowed after results are observed.

**Testing:**
- Write reduced harness tests first and observe failure. Run a tiny deterministic profile twice and require identical scientific rows/summaries despite different replicate batch sizes and execution order.
- Verify paired methods receive the same generated dataset, failures remain rows, replay keys regenerate exact datasets, result schema stays fixed, summary counts match Parquet, and reduced output cannot be accepted as default evidence.
- Test the actual committed-tree inventory expansion: it must contain the validation shell and every selected transitive source, honor only the two exact exclusions, equal the finalized config list, and change the numerical-source digest when any member's copied bytes change. Reject null-sentinel, missing, extra, renamed, dirty, or post-materialization scientific-config changes.
- Inject simulation, inference, calibration, partition writer, hash, assembly, and interruption failures; require the final path to remain nonexistent, transient staging to be cleaned, and only already validated immutable partitions to persist in the distinct checkpoint directory. Resume from those partitions, verify reuse without mutation, and verify final publication copies validated partitions into staging before one atomic rename. Reject checkpoint paths equal to, nested under, or containing the final path, plus overlap/gap/corruption/incomplete/config/core-mismatched partitions and reports.
- Verify the narrow checkpoint ignore rule with `git check-ignore`; final evidence/config/binding paths must not match it. Exercise interruption followed by clean tracked/source preflight and resume from the ignored but independently validated checkpoint directory. Test config/binding mutual exclusion, missing binding, v1 and v2 binding resolution, predecessor/hash mismatch, and no filename/mtime discovery.
- Launch isolated validation subprocesses with x64 disabled/unavailable, wrong platform, wrong full-run seed, or a float64 probe mismatch and require failure before simulation/checkpoint/final staging creation. On success assert float64 simulator leaves, exact configured backend, and identical verified setup in partitions and final report; reject resume under any setup change.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_scripts/test_validate_cis_state.py -x`
- Expected red: reduced orchestration, replay, and artifact assertions fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_scripts/test_validate_cis_state.py tests/test_scripts/test_cis_state_validation_core.py tests/test_sim_cis_state.py`
- Expected green: reduced deterministic validation passes.
- Run: `git diff --check`
- Expected: harness/config/test changes are clean and ready to commit before evidence generation.

**Commit:** `test: add resumable cis-state validation harness`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Generate and commit calibration evidence from a clean source state

**Verifies:** cis-state.AC4.6, cis-state.AC7.1, cis-state.AC7.2, cis-state.AC7.3, cis-state.AC7.4, cis-state.AC7.5

**Files:**
- Generate: `/Users/nicholas/Projects/jaxqtl/.scientific/validation/cis-state/calibration-run-v1/manifest.json`
- Generate: `/Users/nicholas/Projects/jaxqtl/.scientific/validation/cis-state/calibration-run-v1/replicates.parquet`
- Generate: `/Users/nicholas/Projects/jaxqtl/.scientific/validation/cis-state/calibration-run-v1/partitions/`

**Implementation:**
- Commit Tasks 1-3 first and require a clean working tree except the user-owned untracked `state.md`. Compute both that source commit and a canonical SHA-256 over the exact statistical-core source/config files enumerated by the validation schema; the digest excludes generated evidence, later constant-only default binding, and documentation, but includes every simulator, fit, score, calibration, and validation-rule file.
- Run every full-profile partition from that clean source state. Reuse only Task 3's validated immutable partition ranges, then atomically assemble the complete evidence directory. The manifest records source commit and statistical-core digest so the later evidence commit cannot create a circular self-reference.
- Require every prespecified replicate and recovery/edge/observation-level diagnostic. If the gate returns `no-default`, stop and diagnose; do not proceed to Task 7 or weaken/rerun a changed configuration.
- Review the generated summary against the replicate table, then commit the complete evidence directory separately without modifying statistical-core code. Any subsequent change to a file covered by the core digest invalidates this evidence and requires a clean rerun of Tasks 3-4.
- Evidence paths are immutable generations. Version 1 uses the paths listed above and records `generation=1` with null `supersedes_report_path`, `supersedes_report_sha256`, and `supersession_reason`. If a covered correction is required after version 1 has been published, retain version 1, create `calibration-config-v2.json` with the same preregistered scientific settings plus `generation=2` and the exact predecessor path/hash/reason, and publish only to previously nonexistent `calibration-run-v2/`; continue with the smallest unused integer for later corrections. The new report must bind the new clean commit/digest and predecessor fields. Never overwrite, delete, or silently reuse a committed generation. Unit and shell tests must exercise v1 publication, v2 supersession, predecessor-hash mismatch, existing-path refusal, and selection of one explicitly named generation rather than filename-order discovery.

**Verification:**
- Run preflight: `git status --short`
- Expected: no tracked changes; only `/Users/nicholas/Projects/jaxqtl/state.md` may remain untracked.
- Run full evidence: `uv run python scripts/validate_cis_state.py --profile full --config .scientific/validation/cis-state/calibration-config-v1.json --checkpoint-dir .scientific/validation/cis-state/.calibration-run-v1.checkpoints --out .scientific/validation/cis-state/calibration-run-v1`
- Expected: every range is complete and the final manifest records an eligible method, exact source/core/config hashes, and all prespecified gates. A no-default result blocks later default selection.
- Run: `uv run pytest -p no:capture tests/test_scripts/test_validate_cis_state.py tests/test_scripts/test_cis_state_validation_core.py tests/test_sim_cis_state.py`
- Expected: replay and report-validation tests accept the generated artifact.

**Commit:** `test: record cis-state calibration evidence`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Add isolated OneK1K-style benchmark tooling

**Verifies:** cis-state.AC6.1, cis-state.AC6.2, cis-state.AC6.3

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/scripts/cis_state_benchmark_core.py`
- Create: `/Users/nicholas/Projects/jaxqtl/scripts/benchmark_cis_state.py`
- Create: `/Users/nicholas/Projects/jaxqtl/.scientific/validation/cis-state/benchmark-config-v1.json`
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_scripts/test_cis_state_benchmark_core.py` (unit)
- Create: `/Users/nicholas/Projects/jaxqtl/tests/test_scripts/test_benchmark_cis_state.py` (integration/system)

**Implementation:**
- Start `cis_state_benchmark_core.py` with `# pattern: Functional Core`. Colocate immutable closed-schema config/case/report records, exact `smoke`/`onek1k` grid expansion, forbidden-shape and compilation-growth verdicts, completeness/provenance validation, and deterministic solver eligibility/geometric-mean/tie-break recommendation. This pure policy is shared by the harness, report validation, and Phase 7 evidence binding; it performs no subprocess, timing, resource, JAX-device, or filesystem work.
- Start `benchmark_cis_state.py` with `# pattern: Imperative Shell` and a standard-library-only parent/child bootstrap: parse the closed config or binding and set pinned platform/x64 environment before importing JAX/jaxqtl in either process. Require parent and every fresh case child to verify `jax.config.x64_enabled`, exact backend, and an actual float64 probe before fixture generation or timing. Then generate stable synthetic inputs, gather timing/resource/JAX metrics, pass bounded scalar/shape/setup records back through core verdicts, and return nonzero for incomplete cases or rejected verdicts. Do not copy the existing unreachable child-metrics code path; write child metrics after work and before returning.
- Accept mutually exclusive `--config` and `--binding`: direct evidence generation uses the explicit benchmark config, while the final audit resolves the explicitly selected benchmark generation from Task 7's canonical selected-evidence JSON after validating all binding/config/report/source/harness hashes. Never select by filename order or mtime.
- Make `benchmark-config-v1.json` a closed schema with no runtime defaults: `generation=1`, null supersession fields, root seed `20260814`, CPU-only execution, JAX x64 enabled, float64 arrays, one cold subprocess per case, one explicit lower/compile/first-execution sequence, and exactly seven synchronized warm executions summarized by min/median/max. Pin candidate tuples to PROPACK `(tol=1e-7,maxiter=1000,seed=20260814,ncv=null)` and ARPACK `(tol=1e-7,maxiter=1000,seed=20260814,ncv=min(min(M,q_active)-1,max(2*r+1,20)))`, rejecting a case if that resolved `ncv` is not strictly between `r` and `min(M,q_active)`. Pin the per-gene absorbed fit to `QRSolve`, `min_informative_donors=30`, mean/dispersion iteration caps `100/50`, predictor/coefficient relative tolerances `1e-7`, `delta_log_phi=1e-6`, phi bounds `[1e-8,1e3]`, `20` backtracking steps, `rank_rtol=null` resolving to `max(M_active,d)*eps(float64)`, normal-equation tolerance `1e-8`, and Cholesky accuracy budget `1e-8`. Pin variant block size `64`, capacities `(128,512,1024)`, multiplier `(num_replicates=1999,replicate_block_size=64,interpretation_threshold=null)`, and moment `(num_probes=512,probe_block_size=64,approximation=three-moment,invalid_three_moment=fallback-two-moment,validity_rtol=1e-10)`.
- Pin the OneK1K per-gene grid from the supplied model: `N=989`, `M in {50_000,100_000}`, `p in {100,300,1_000}`, and `r in {20,50,100}`. For state construction use both cell counts, `q=20_000`, an explicit sparse `500` expected nonzeros per cell, ranks `20/50/100`, and one full 22-chromosome LOCO case at `M=50_000`, `N=989`, `q=20_000`, and `r=50` in addition to single-exclusion/operator cases. Generate sparse fixtures directly without a dense `M x q` intermediate and exclude fixture generation from timed regions.
- Pin fixture generation in that config. Define a solver-independent `fixture_id=(workload_kind,M,N,p,r,q,exclusion,fixture_replicate)` and derive its key by folding a canonical hash of that tuple into the root key; define a separate `case_id=(fixture_id,solver,case_replicate)` for subprocess/timing identity. Record a canonical fixture payload hash, and require paired ARPACK/PROPACK cases to have identical `fixture_id`, fixture hash, tolerance, iteration control, and resolved inputs while retaining distinct case IDs. For state cases set `donor_index[c]=c mod N`; choose exactly 500 distinct gene indices per cell uniformly without replacement; assign each stored count as `1 + Poisson(1.5)`; and set gene chromosome to `1 + (gene_index mod 22)`. For per-gene cases also set `donor_index[c]=c mod N`, so donors `0:(M mod N)` have `ceil(M/N)` cells and the rest have `floor(M/N)` cells in cyclic cell order. Draw `G_ij ~ Binomial(2,0.30)`, natural `S_ck ~ Normal(0,1)`, three donor and two cell nuisance covariates independently standard normal, donor intercepts `Normal(-4.6,0.4^2)`, and `offset_c=log(1000)+0.3*z_c`. For every benchmark rank define `beta_S[k]=0.2*(-1)^k/(k+1)` and `theta[a,k]=0.1*(-1)^(a+k)/((a+1)*(k+1))` for integer indices `0 <= k < r` and `0 <= a < 3`, with `beta_C=(0.12,-0.08)`; draw NB2 outcomes at `phi=0.2` and zero interaction. Do not adapt distributions after timing or replace invalid cases; report them as failures.
- Benchmark PFlog statistics/operator actions, donor centering/balancing, truncated factorization, and LOCO iteration separately. Benchmark absorbed NB fitting, donor-state aggregation/statistic, nuisance correction, moment calibration, and multiplier calibration separately for every per-gene grid case.
- On JAX kernels, lower and compile fixed-shape callables explicitly, synchronize results with `block_until_ready`, and report lowering, compilation, first execution, and median/min/max of exactly seven warm executions. Run with cold in-process caches per case and enable cache-miss explanations for the bucket audit; do not infer compilation time merely as cold minus warm.
- Normalize `resource.RUSAGE_SELF.ru_maxrss` units by platform for host peak RSS. When supported, record JAX device `memory_stats()` and an optional device-memory profile; label live snapshots separately from peak. Record input/storage bytes and solver iteration/residual diagnostics.
- Audit logical JAX intermediate shapes from lowered jaxprs and wrap host/SciPy operator boundaries to record input/output/temporary shapes. Hard-fail on `M x M`, `M x N`, `M x p`, `M x (p*r)`, `N x (p*r)`, full `T`, or `N x N` covariance allocation. Report the largest observed shapes and the bounded block envelope rather than claiming XLA physical allocations are fully observable.
- Feed multiple genes through repeated capacities and require compilation/cache-miss growth no greater than the unique tuple of configured variant bucket, state rank, nuisance dimension, calibration method/config, dtype, and platform. Do not call `jax.clear_caches` between genes in this audit.
- Compare ARPACK and PROPACK only on identical state fixtures and tolerances. A solver is eligible only if every case converges with Phase 2 residual/orthogonality thresholds and no prohibited shape. If both qualify, recommend the lower geometric-mean wall-time solver; break a within-10% tie by lower peak RSS, then stable lexical name. Keep `--solver` required if neither is eligible; benchmark speed never affects calibration selection.

**Testing:**
- Add pure benchmark-policy and smoke-shell tests first and observe failure.
- In `test_cis_state_benchmark_core.py`, verify closed-schema rejection of every omitted/unknown field, every complete absorbed/multiplier/moment config value, explicit null interpretation threshold, exact values and deterministic fixture replay, exact candidate-tuple and ARPACK-`ncv` resolution, strict truncated-rank rejection for both solvers, exact grid expansion, seven-warm-run completeness, report completeness, every forbidden/allowed shape verdict, bucket compilation-growth accounting, solver eligibility, geometric means, the within-10% memory tie, lexical tie-break, and no recommendation from partial/ineligible cases. Across all ranks, assert exact `beta_S (r,)`, `theta (3,r)`, finite predictor components, and deterministic NB2 outcomes before timing; across both `M` values, assert cyclic donor coverage and exact floor/ceiling counts. Paired ARPACK/PROPACK cases must have distinct case IDs but identical solver-independent fixture IDs/hashes and numerical controls. Changing any config field after report generation or mutating one byte in any expanded numerical-source or benchmark-harness inventory member must invalidate eligibility.
- In `test_benchmark_cis_state.py`, verify subprocess isolation, RSS unit normalization, synchronization before timing, compile/warm separation, stable bounded records, and nonzero exit on incomplete/OOM/convergence/forbidden-shape cases. Feed synthetic jaxprs/operator records through the shell-to-core boundary and exercise bucket reuse plus intentional static-shape changes. Test atomic report success, rejection of an existing destination, and injected interruption/write/validation failures leaving no final file and cleaning the sibling staging file.
- In isolated parent/child cases, force disabled/unavailable x64, wrong backend, or float64-probe mismatch and require nonzero exit before fixture/timing/staging with no final report. Successful cases assert all real fixture leaves are float64 and every child setup record exactly matches the config; a single mismatch makes the report incomplete/ineligible.
- Run only `smoke` in pytest; mark the OneK1K profile as an explicit non-CI command.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_scripts/test_cis_state_benchmark_core.py tests/test_scripts/test_benchmark_cis_state.py -x`
- Expected red: benchmark grid, timing, memory, and allocation-audit assertions fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_scripts/test_cis_state_benchmark_core.py tests/test_scripts/test_benchmark_cis_state.py`
- Expected green: smoke benchmark contract tests pass.
- Run: `git diff --check`
- Expected: benchmark harness and smoke tests are clean and ready to commit before the measured run.

**Commit:** `bench: add OneK1K-scale cis-state benchmark harness`
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: Generate and commit the OneK1K benchmark report

**Verifies:** cis-state.AC6.1, cis-state.AC6.2, cis-state.AC6.3

**Files:**
- Generate: `/Users/nicholas/Projects/jaxqtl/.scientific/validation/cis-state/benchmark-report-v1.json`

**Implementation:**
- Commit Task 5 first and require a clean working tree except `state.md`. Record the clean source commit, benchmark-config hash, environment, and the same broad canonical numerical-source digest used for calibration provenance—therefore including every state, distribution, inference, score, and calibration implementation being measured—before starting subprocess cases; separately hash `scripts/cis_state_benchmark_core.py`, `scripts/benchmark_cis_state.py`, and the benchmark config as the benchmark-harness inventory used to interpret the measurements.
- Run the pinned OneK1K profile without changing dimensions after observation. Refuse an existing final report. Write the completed JSON to a unique sibling staging file, close it, validate report completeness/provenance/case hashes/solver eligibility/allocation/compilation/timing fields through `cis_state_benchmark_core.py`, then atomically `os.replace` the staging file onto the previously nonexistent final path. Clean staging on any failure and do not claim filesystem crash durability.
- If resources cannot execute the pinned profile, any case fails, publication is interrupted, or core validation fails, report the phase blocked and leave no final report. Do not shrink the grid, relabel smoke evidence, recommend a solver from partial cases, or publish a success report.
- Benchmark reports also use immutable generations. Version 1 has `generation=1` and null supersession fields. A covered correction after publication retains version 1, creates `benchmark-config-v2.json` with unchanged benchmark policy unless a separately reviewed prespecification change is required, records the version-1 report path/hash/reason, and publishes only to previously nonexistent `benchmark-report-v2.json`; later corrections increment monotonically. Tests cover v2 predecessor validation, explicit-generation binding, existing-target refusal, and the fact that neither lexical filename order nor mtime selects evidence.

**Verification:**
- Run preflight: `git status --short`
- Expected: no tracked changes; only `/Users/nicholas/Projects/jaxqtl/state.md` may remain untracked.
- Run OneK1K: `uv run python scripts/benchmark_cis_state.py --profile onek1k --config .scientific/validation/cis-state/benchmark-config-v1.json --out .scientific/validation/cis-state/benchmark-report-v1.json`
- Expected: every case completes, reports cold/compile/warm/RSS/shapes, and has zero prohibited allocation or unexpected compilation-growth findings.
- Run: `uv run pytest -p no:capture tests/test_scripts/test_cis_state_benchmark_core.py tests/test_scripts/test_benchmark_cis_state.py`
- Expected: report-schema and provenance tests accept the generated report.

**Commit:** `bench: record OneK1K-scale cis-state evidence`
<!-- END_TASK_6 -->

<!-- START_TASK_7 -->
### Task 7: Apply the evidence-selected defaults

**Verifies:** cis-state.AC4.6, cis-state.AC7.5

**Files:**
- Create: `/Users/nicholas/Projects/jaxqtl/.scientific/validation/cis-state/selected-evidence.json`
- Create: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/_cis_state_defaults.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/hypothesis/__init__.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_state_artifact_contract.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/io/_cis_state_contract.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/map/cis_state.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/src/jaxqtl/cli.py`
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cluster_calibration.py` (unit)
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_io/test_state_artifact.py` (unit/integration)
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_io/test_cis_state_io.py` (integration)
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cis_state_workflow.py` (integration)
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cli/test_state_cli.py` (integration)
- Modify: `/Users/nicholas/Projects/jaxqtl/tests/test_cli/test_cis_state_cli.py` (integration)

**Implementation:**
- Load and validate one explicitly named completed evidence generation against its committed config/hash, predecessor chain, and the exact gate from Task 2. Do not discover a report by filename order or mtime, and do not proceed if its decision is `no-default`, incomplete, reduced, superseded by the explicitly selected generation, or generated from a different code/config/environment contract.
- Write canonical `selected-evidence.json` as the source-tree implementation/audit binding. It contains schema version, the explicitly selected calibration and benchmark generation numbers, config/report paths and hashes, predecessor path/hash fields, numerical-source digest, benchmark-harness digest, and selection reason. During Task 7 generation, validate this file and embed its resolved values as self-contained constants in `_cis_state_defaults.py`; the installed package never reads `.scientific/` at import or runtime, because current wheels/sdists contain only package sources. Updating to v2+ is an explicit reviewed edit that regenerates both the JSON and embedded module, with equality tests preventing them from becoming independent selection authorities.
- Start `_cis_state_defaults.py` with `# pattern: Functional Core`. This small module is justified as a contract-critical evidence binding deliberately outside the hashed statistical algorithms: pin `DEFAULT_CIS_STATE_CALIBRATION`, its complete config, `DEFAULT_CIS_STATE_VALIDATED_LEVELS=(0.01,0.05)`, report/config/statistical-core SHA-256 values, and a stable accessor. Preserve both explicit Phase 5 calibration APIs and the explicit `test_cis_state_interaction` array API unchanged; direct array callers obtain and pass the exported default contract when desired.
- Allow the `cis-state` CLI to omit calibration/config only after these constants exist. With omission, use the evidence-selected method and its entire exact config; reject all method-specific calibration flags so no partial merge is possible. With explicit `--calibration`, require the complete matching Phase 6 option set and ignore no supplied field. The common `--interpretation-threshold` remains method-independent with default `0.05` and overrides only the declared interpretation/warning layer, never bootstrap draws, moment probes, or the evidence config.
- CLI help, result rows, stderr warnings, and manifests state the resolved method/config, replicate/probe count, resolution where applicable, declared interpretation threshold, validated levels `{0.01,0.05}`, warning fields, and evidence report hash. Define a total level mapping over `(0,1)`: `none` at exactly `0.01` or `0.05`, `unvalidated_level_below` below `0.01`, `unvalidated_level_between` strictly between the validated levels, and `unvalidated_level_above` above `0.05`. Store bootstrap resolution adequacy independently as `calibration_resolution_warning = (method is multiplier and threshold < 1/(B+1))`, so simultaneous warnings compose without precedence; it is false for moment calibration. Summarize each nonzero warning class on stderr once per run. A default-derived p-value is retained, but interpretation at an unvalidated level is labeled extrapolation. A threshold below `0.01` is a usage error with the evidence default and requires explicit calibration plus a configuration whose bootstrap resolution or documented reference precision is at least as fine as the threshold.
- If the benchmark produced an eligible state-factor recommendation, validate its source/config/core/harness hashes and pin the complete benchmarked CLI solver tuple—not only its name—including solver, tolerance, iteration control, seed, and the shape-resolved ARPACK `ncv` formula when applicable, together with the benchmark-report hash. Then the `state-factor` CLI may omit that tuple only as a unit; an explicit solver requires its complete matching Phase 2 controls and never inherits selected-default fields piecemeal. Extend state manifest schema version 1 before release with `solver_selection_source` (`explicit` or `benchmark-default`) plus nullable benchmark report/config/core-harness hashes: default-selected tuples record all hashes and the resolved numeric `ncv` for each artifact, explicit tuples record source `explicit` and null hashes, and no-eligible-solver runs remain explicit. Preserve the Phase 2 library function's required explicit solver/config.
- If an evidence-selected ARPACK default is inapplicable because `rank=min(M,q_active)-1` leaves no integer satisfying `rank < ncv < min(M,q_active)`, fail after shape validation with runtime status `1`, create no artifact, and tell the caller to supply a complete explicit PROPACK tuple. Help documents this default-applicability limit. Never lower rank, alter `ncv`, or silently fall back to PROPACK; explicit invalid ARPACK configurations retain the same actionable failure boundary.
- Never regenerate or weaken evidence in this task. Any code change to a file enumerated by the numerical-source digest invalidates the selected report and requires the immutable next-generation Tasks 3-4 workflow. The new constants/accessor and CLI wiring are excluded from that predeclared digest and may not alter numerical algorithms. Bind the selected calibration and benchmark generation numbers, paths, predecessor hashes, and report hashes into constants/manifests so supersession is auditable.

**Testing:**
- Add evidence-binding tests first and observe failure.
- Assert constants/config/hash/generation/predecessor fields match the explicitly selected committed full report, the accessor-fed explicit array invocation and CLI default agree exactly, manifest/help expose the evidence binding, the original array API remains explicit, and invalid/reduced/no-default or implicitly discovered reports cannot configure the API. Include valid v1 and v2 fixtures and prove that the binding changes only when the named generation is updated.
- Build both wheel and sdist into a test-owned temporary directory, install the wheel into an isolated temporary environment, change to a directory without the source checkout, and require imports plus `state-factor`/`cis-state` default help and parser resolution to work without `.scientific/`. Inspect the sdist to confirm the binding JSON is not accidentally required or packaged; embedded constants must equal the source-tree binding at build time.
- Assert both audit shells resolve the config paths from `selected-evidence.json`; a v2 fixture must drive v2 reduced/smoke inputs even when valid v1 files remain and have newer mtimes. Reject binding/default-module disagreement, unknown binding fields, missing predecessor hashes, and path/hash drift.
- With an ARPACK-selected default, test the largest ordinary applicable rank and the strict-rank edge with no legal `ncv`; the edge must return status `1`, emit actionable PROPACK guidance, and leave no manifest/artifact. An explicit complete PROPACK tuple succeeds on the same shape, and no test may observe an automatic solver switch.
- Test both possible decision fixtures (moment and multiplier), omitted-versus-explicit complete-config precedence, rejection of partial/mismatched method flags, selected method resolution/config/validated-level set, exact level-code plus independent resolution-warning row/manifest/stderr behavior at `0.03` with adequate/inadequate multiplier resolution and above `0.05`, both methods, rejection of a below-`0.01` evidence-default threshold, explicit adequately resolved overrides, and complete solver-tuple selected/explicit/retained-required manifest provenance. Assert no timing field can alter calibration defaults.

**Verification:**
- Run red: `uv run pytest -p no:capture tests/test_cluster_calibration.py tests/test_io/test_state_artifact.py tests/test_io/test_cis_state_io.py tests/test_cis_state_workflow.py tests/test_cli/test_state_cli.py tests/test_cli/test_cis_state_cli.py -x`
- Expected red: default/evidence-binding assertions fail before implementation.
- Run green: `uv run pytest -p no:capture tests/test_cluster_calibration.py tests/test_io/test_state_artifact.py tests/test_io/test_cis_state_io.py tests/test_cis_state_workflow.py tests/test_cli/test_state_cli.py tests/test_cli/test_cis_state_cli.py tests/test_scripts/test_cis_state_validation_core.py`
- Expected green: defaults match the immutable report and overrides remain valid.

**Commit:** `feat: select validated cis-state defaults`
<!-- END_TASK_7 -->

<!-- START_TASK_8 -->
### Task 8: Document the workflow, estimand, evidence, and limitations

**Verifies:** cis-state.AC6.5

**Files:**
- Modify: `/Users/nicholas/Projects/jaxqtl/.gitignore`
- Create: `/Users/nicholas/Projects/jaxqtl/docs/guide/cis_state.md`
- Create: `/Users/nicholas/Projects/jaxqtl/docs/api/data/state.md`
- Create: `/Users/nicholas/Projects/jaxqtl/docs/api/inference/cell_nb.md`
- Create: `/Users/nicholas/Projects/jaxqtl/docs/api/hypothesis/cis_state.md`
- Create: `/Users/nicholas/Projects/jaxqtl/docs/api/mapping/cis_state.md`
- Create: `/Users/nicholas/Projects/jaxqtl/docs/api/simulation/cis_state.md`
- Modify: `/Users/nicholas/Projects/jaxqtl/zensical.toml`
- Review: `/Users/nicholas/Projects/jaxqtl/AGENTS.md` and companion `CLAUDE.md` files if present after implementation

**Implementation:**
- Keep the existing broad generated-API ignore, but add ordered negative rules that unignore the five exact parent directories and exact planned Markdown files (`data/state.md`, `inference/cell_nb.md`, `hypothesis/cis_state.md`, `mapping/cis_state.md`, `simulation/cis_state.md`). Do not unignore arbitrary generated API output.
- Add the guide and API pages, including the simulation reference, to the existing navigation without reorganizing established pages. Use public signatures and actual generated manifests/help as sources; do not promise AnnData/TSV metadata, arbitrary covariance, variant/state weights, kinship, or a future sparse accelerator.
- Require repository-style raw docstrings for every new exported state, I/O, inference, hypothesis, mapping, defaults, and simulation entrypoint/container, with Arguments/Returns and Raises or Failure Modes plus determinism/backend notes where relevant. Each API page must resolve its explicit mkdocstrings symbol list under the strict build.
- State the exact NB2 null, absorbed donor fixed effects, required donor-covariate-by-state nuisance interactions, absent genotype main effect, omnibus kernel scaling, efficient donor contributions, and selected calibration.
- Explain that the estimand is the within-donor relationship between state and focal-gene expression modified by cis genotype; factors are natural unwhitened LOCO factors learned with donor balancing but used unweighted in the model.
- Document LOCO PFlog alpha, fixed-state conditional interpretation, same-experiment factor estimation not included in p-value uncertainty, independent-donor assumption, kinship out of scope, population adjustment requirements, zero-total/informative-donor behavior, failure statuses, deterministic keys, variant buckets, bootstrap resolution, moment trace SE meaning, and explicit overrides.
- Include exact initial file schemas, rows-cells/columns-genes orientation, Parquet-only metadata/covariates, state/run artifact layouts, input hashes/reconciliation, CLI exit codes, examples for state-factor and cis-state, and recovery/replay instructions.
- Summarize the versioned calibration and benchmark reports with their hashes, nominal levels/replicates/tolerances, observed rejection rates, selected method/reason, solver recommendation, timings/memory, and unresolved concerns. State prominently that default calibration was validated only at interpretation levels `0.01` and `0.05`, not continuously between them; outputs below `0.01` are not evidence of genome-wide extreme-tail calibration, and smaller thresholds require an explicit adequately resolved calibration plus their own validation. Link to the tracked reports rather than copying machine-specific detail selectively.
- Apply project-context maintenance after all source changes: detect whether the repository uses root `AGENTS.md` or `CLAUDE.md`, read current files, and update only contract-owning context when it exists and is stale. The repository had no tracked context file during planning, so do not invent a new context system solely for this feature; record that audit if none exists.

**Testing:**
- Add no prose-snapshot tests. Validate every CLI example against `--help`/parser fixtures and ensure mkdocstrings resolves the exported public objects.
- Search documentation for prohibited claims (`arbitrary Omega`, kinship support, unconditional factor uncertainty, state/variant weights) and for stale required-calibration text after default selection.

**Verification:**
- Run: `git check-ignore -q docs/api/data/state.md docs/api/inference/cell_nb.md docs/api/hypothesis/cis_state.md docs/api/mapping/cis_state.md docs/api/simulation/cis_state.md`
- Expected: exit `1`; none of the five authored API pages is ignored.
- Stage the exact documentation paths and run: `git ls-files --stage docs/api/data/state.md docs/api/inference/cell_nb.md docs/api/hypothesis/cis_state.md docs/api/mapping/cis_state.md docs/api/simulation/cis_state.md`
- Expected: all five pages are present in the Git index before the docs commit.
- Run: `uv sync --frozen --extra dev --extra docs`
- Expected: the locked development and documentation environment installs without changing the lock.
- Run: `uv run zensical build --strict --clean`
- Expected: strict documentation build passes with no missing API reference or navigation path.
- Run: `uv run pytest -p no:capture tests/test_cli/test_state_cli.py tests/test_cli/test_cis_state_cli.py`
- Expected: documented command surfaces match parser behavior.

**Commit:** `docs: document cis-state analysis`
<!-- END_TASK_8 -->

<!-- START_TASK_9 -->
### Task 9: Complete the branch-wide scientific and engineering audit

**Verifies:** cis-state.AC1.1-cis-state.AC1.7, cis-state.AC2.1-cis-state.AC2.6, cis-state.AC3.1-cis-state.AC3.6, cis-state.AC4.1-cis-state.AC4.6, cis-state.AC5.1-cis-state.AC5.6, cis-state.AC6.1-cis-state.AC6.5, cis-state.AC7.1-cis-state.AC7.5

**Files:**
- Review all files changed since `/Users/nicholas/Projects/jaxqtl/.plans/design-plans/2026-08-14-cis-state.md` was approved.

**Implementation:**
- Confirm full evidence and OneK1K reports are complete, hash-bound to the implemented code/config, and accurately summarized. If a covered change occurred before an evidence generation was published, discard only unpublished staging/checkpoints and rerun that generation. If a covered correction occurs after publication, keep the prior committed generation, create the next versioned config/report paths with predecessor hash and reason, rerun into those previously nonexistent paths, update the explicit default binding, and commit the superseding evidence; never overwrite published evidence.
- Review optimized algebra against every explicit dense oracle. Do not weaken tolerances or tests to conceal a discrepancy; stop and diagnose any factor, absorbed-fit, score, contribution, calibration, or simulation mismatch.
- Audit source module cohesion and FCIS classifications. Merge trivial leaf files if implementation created them without a stable responsibility; preserve the `state`, `infer`, `hypothesis`, `io`, and `map` responsibility boundaries.
- Search source/jaxprs/allocation reports for prohibited dense shapes and Python cell/donor loops, verify bucket-bounded compilation behavior, and ensure no per-gene `jax.clear_caches` workaround was introduced.
- Confirm donor covariates remain required, state/variant weights remain absent, input formats match the initial contract, all failures are explicit, and existing pseudobulk APIs/commands/outputs remain unchanged.
- Preserve `/Users/nicholas/Projects/jaxqtl/state.md` as user-owned untracked/unstaged. Run the canonical code-review and context-maintenance workflows before final integration.

**Verification:**
- Run reduced validation in a fresh directory using the explicitly selected generation: `zsh -c 'audit_tmp=$(mktemp -d /tmp/jaxqtl-cis-state-reduced.XXXXXX); exec uv run python scripts/validate_cis_state.py --profile reduced --binding .scientific/validation/cis-state/selected-evidence.json --checkpoint-dir "$audit_tmp/checkpoints" --out "$audit_tmp/final"'`
- Expected: deterministic reduced validation completes and is labeled non-evidentiary.
- Run smoke benchmark in a fresh directory using the explicitly selected generation: `zsh -c 'audit_tmp=$(mktemp -d /tmp/jaxqtl-cis-state-benchmark.XXXXXX); exec uv run python scripts/benchmark_cis_state.py --profile smoke --binding .scientific/validation/cis-state/selected-evidence.json --out "$audit_tmp/report.json"'`
- Expected: timing/memory/shape schema completes with no prohibited allocation.
- Run: `uv run pytest -p no:capture`
- Expected: the complete suite passes with no new warnings or regressions.
- Run: `uv run ruff check src tests scripts`
- Expected: no lint errors.
- Run: `uv run ruff format --check src tests scripts`
- Expected: all files already formatted.
- Run: `uv run ty check src tests scripts`
- Expected: no type errors.
- Run: `uv run zensical build --strict --clean`
- Expected: strict documentation build passes.
- Run: `git diff --check`
- Expected: no whitespace errors.
- Recheck the committed full evidence and OneK1K reports rather than silently rerunning a different configuration. Report branch, files, exact model, gates, empirical rejection/power/convergence, selected calibration/reason, benchmark time/memory/shapes, solver decision, and unresolved concerns.

**Commit:** None unless the audit finds a required fix; repeat its owning task, apply the unpublished-rerun or immutable next-generation evidence workflow above as appropriate, and make narrow correction/evidence/default-binding commits.
<!-- END_TASK_9 -->
