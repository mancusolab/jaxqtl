# Cis-by-state scientific kickoff

- mode: `provided-model`
- model_path_decided: yes
- codebase_investigation_complete_if_port: n/a
- simulation_contract_complete_if_in_scope: yes

## Model sources

- `state.md`: user-provided scientific objective, model equations, computational constraints, validation scenarios, and completion criteria.
- User-confirmed design decisions from the 2026-08-14 brainstorming session:
  - treat the model and scientific requirements as fixed;
  - use a parallel vertical-slice architecture;
  - start with sparse CSR/CSC `.npz` counts and separate metadata tables;
  - use CPU sparse state construction and JAX CPU/GPU inference;
  - require donor covariates and include donor-covariate-by-state nuisance terms;
  - estimate PFlog dispersion separately for each LOCO chromosome by cached sufficient-statistic subtraction;
  - standardize genotype columns and fix both interaction-weight vectors to one;
  - do not expose variant or state weights in the initial public API.

## Inference and solver rules

- Fit the supplied NB2 log-link null with absorbed donor fixed effects.
- Reuse the existing negative-binomial family, link, likelihood, dispersion, and small dense solve machinery where their contracts fit.
- Use a custom grouped IRLS shell because the existing GLM assumes a dense observation-level design and covariance matrix.
- Use SciPy sparse arrays and a `LinearOperator`-compatible truncated SVD for the initial CPU state-factor backend.
- Use existing Lineax-backed QR or Cholesky solves for reduced nuisance systems; do not materialize donor dummies or interaction designs.
- Return structured per-gene status for recoverable numerical failures. Raise Python exceptions for structural boundary errors before JIT.

## Known mathematical and numerical risks

- Automatic PFlog estimates can be nonpositive or degenerate after chromosome exclusion; fail with diagnostics and require an explicit override rather than clipping.
- Zero-total donors create boundary donor intercepts and must be removed before the null fit.
- The donor-cluster moment approximation may be inaccurate in extreme tails; its default status depends on simulation against multiplier calibration.
- A finite multiplier bootstrap has resolution `1 / (B + 1)` and cannot support thresholds below that resolution.
- State factors are treated as fixed downstream. Related donors and kinship are outside this branch.

## Simulation contract

Simulation-based validation is in scope. A deterministic JAX simulation entrypoint will generate all arrays consumed by the inferential API:

```python
simulate_cis_state(
    key,
    params,
    config,
) -> CisStateSimulation
```

The result contains raw focal-gene counts, donor genotypes, cell states, donor indices, offsets, cell covariates, required donor covariates, true means and dispersion, the true interaction matrix, and scenario metadata. PRNG keys are explicit and split deterministically. Planned experiments cover the exact NB null, strong overdispersion, unequal library sizes and donor cell counts, extra within-donor dependence, population/state confounding, zero-count donors, and increasing interaction variance under alternatives.
