# Failures and convergence

jaxQTL distinguishes rows that represent valid association results from fits or tests that did not produce an
interpretable result.

## Cis validity

Use `result_valid` as the first filter for cis results. When it is false, `failure_reason` records why no association
was selected. The current explicit reason is `no_finite_pvalues`.

`model_converged` describes the selected model fit. For Beta-permutation calibration, `perm_converged` requires both
the calibration estimate and fitted Beta parameters to converge.

!!! warning "Convergence does not establish model adequacy"

    A converged optimizer only indicates that its numerical stopping rule was met. It does not show that the response
    family, covariates, offset, or asymptotic test is appropriate for the data.

## Skipped phenotypes

Cis and nominal scans skip phenotypes with no variants in the requested window or with zero or NaN phenotype
variance. If every phenotype is skipped, the mapper yields an empty result frame and logs a warning.

Trans mapping removes phenotypes with zero or NaN variance before testing. If none remain, it produces no result
blocks.

## Logs

Use `--verbose` for per-region progress messages. Runtime exceptions from input readers or model fitting are not
converted into result rows; they stop the command and should be resolved from the reported error.
