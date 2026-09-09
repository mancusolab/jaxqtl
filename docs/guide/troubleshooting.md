# Troubleshooting

jaxQTL distinguishes rows that represent valid association results from fits or tests that did not produce an
interpretable result.

## Cis validity

Use `result_valid` as the first filter for cis results. It indicates that the scan selected a variant with a finite
nominal p-value. When it is false, `failure_reason` records why no association was selected. The explicit reason
`no_finite_pvalues` means every SNP-level p-value in the window was nonfinite.

`model_converged` describes the selected model fit. For Beta-permutation calibration, `perm_converged` requires both
the calibration estimate and fitted Beta parameters to converge.

Check that `pvalue_adj` is finite even when `result_valid` is true. A selected lead can coexist with failed
Beta calibration or a nonfinite ACAT aggregate. The separate fields should not be interpreted as a single
overall success flag.

## Interpreting an unsuccessful fit

`model_converged = false` can indicate an iteration limit, exhausted backtracking, or an unchanged state that
still fails the gradient criterion. Increasing `--max-iter` only addresses the first of these conditions.

Inspect phenotype counts, offsets, and covariates before changing solver settings. Very sparse phenotypes,
redundant covariates, and covariate groups with no expression can leave parameters poorly determined. Compare
negative log likelihood only for the same observations, response family, and model specification.

Model convergence, successful tail approximation, and successful gene-level calibration are separate properties.
See [Post-process cis results](postprocessing.md) for filtering examples.

## Stopping rules

Score and SPA tests fit a covariate-only null model; GLM Wald tests fit each variant's full model. These GLM fits
share the controls below. Gaussian linear models use a direct least-squares solve.


GLM convergence requires both of the following:

1. The absolute change in **total** negative log likelihood is at most `--tol`.
2. The scaled gradients at the accepted fit satisfy `--gtol`.

The coefficient gradients are divided by the sample count and each design column's root-mean-square magnitude.
NB2 also checks the per-observation dispersion gradient, accounting for feasible movement at a dispersion bound.
A small likelihood change alone, or a dispersion estimate at a bound, does not establish convergence.

Both tolerances default to `1e-3`. `--max-iter` defaults to `1000`, and the initial trial `--step-size` is `1.0`.
These settings also govern the Poisson fit used to initialize NB2. The exact gradient criteria are given in the
[model API reference](../api/models/glm.md#convergence).

To examine sensitivity to a tighter gradient criterion, rerun the same inputs with an explicit setting:

```bash
jaxqtl cis \
  --bfile tutorial/input/chr22_N100 \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --covar tutorial/input/donor_features.tsv \
  --gene-list tutorial/input/genelist_10 \
  --model nb \
  --test score \
  --set-offset-from-libsize \
  --acat \
  --spa \
  --gtol 1e-6 \
  --out tutorial/output/cis_tight
```

A smaller gradient tolerance asks for a more stationary fit and can require more iterations. Compare convergence
flags, dispersion estimates, and association results as well as runtime. `--tol` and `--gtol` do not set the
saddlepoint root-solver or permutation-calibration tolerances.


!!! warning "Check model adequacy before interpreting discoveries"

    A converged optimizer only indicates that its numerical stopping rule was met. It does not show that the response
    family, covariates, offset, or asymptotic test is appropriate for the data.

## SPA and aggregation

SPA uses the normal approximation when its score cutoff or support checks do not request a correction. It also
falls back to the normal approximation if root finding fails or the correction is invalid. The output does not
include a separate SPA-success flag; `model_converged` describes the fitted response model.

**We strongly recommend `--spa --acat` for score-test ACAT scans** because inaccurate variant tail probabilities
can disproportionately affect the aggregate. SPA's normal-approximation fallback still applies.

ACAT propagates nonfinite input p-values. Its Cauchy transform is sensitive to p-values near both zero and one;
a gene-level p-value near one can occur even when some nominal p-values are small. Inputs containing both exact
zero and exact one raise an error.

Beta permutation calibrates statistics against the same testing procedure under permutation, so distortions
shared by observed and permuted tests can enter the reference distribution instead of being passed directly
to ACAT as nominal p-values. This does not protect against failed fits or failed calibration.
For Beta permutation, filter on `perm_converged` and a finite adjusted p-value. A failed reference estimate or
Beta fit can leave `pvalue_adj` nonfinite. Increasing the GLM iteration limit does not change calibration's own
iteration limits.

## Skipped phenotypes

Cis and nominal scans skip phenotypes with no variants in the requested window or with zero or NaN phenotype
variance. If every phenotype is skipped, the mapper yields an empty result frame and logs a warning.

Trans mapping removes phenotypes with zero or NaN variance before testing. If none remain, it produces no result
blocks.

## Logs

Use `--verbose` for per-region progress messages. Runtime exceptions from input readers or model fitting are not
converted into result rows; they stop the command and should be resolved from the reported error.
