# Tests and gene-level calibration

jaxQTL separates the response model, variant-level test, and gene-level calibration method. These choices answer
different questions, but tail calibration is particularly important when aggregating variant p-values.

See [Statistical model](model.md) for response families and effect interpretation.

## Variant-level tests

The score test fits one covariate-only model per phenotype and tests each genotype against that null fit. It is the
default for cis scans because the null-model work is reused across variants.

The Wald test fits the tested variant coefficient and reports its estimate, standard error, and Wald statistic. It is
the usual choice for nominal results when effect estimates for every variant are required.

!!! warning "Robust standard errors apply only to Wald tests"

    `--robust-se` selects Huber–White sandwich standard errors for Wald inference. It does not make the implemented
    score statistic or saddlepoint approximation misspecification-robust.

## Tail and gene-level calibration

`--spa` attempts a saddlepoint approximation for score statistics with absolute normal z-score above 1.96 that
pass the CGF's score-support checks. It uses the fitted null model to evaluate the tail and can be useful when
normal approximations are inaccurate, such as at low minor-allele counts. Smaller statistics use the normal tail.

If the saddlepoint root solve fails or its correction is invalid, the implementation returns the normal
approximation. A finite p-value or `model_converged = true` does not identify whether SPA succeeded. See
[Troubleshooting](troubleshooting.md).

Gene-level testing does not require permutations. `jaxqtl cis --spa --acat` combines SPA-calibrated score-test
p-values with the aggregated Cauchy association test (ACAT). Without `--acat`, the CLI uses permutation
calibration and a fitted Beta approximation.

### Faster cis scans with SPA and ACAT

SPA + ACAT is typically substantially faster than permutation scans because it fits each gene's null model once
and avoids repeatedly fitting and testing shuffled phenotypes. The speed difference depends on the data and the
number of permutations used for comparison.

See the [Quickstart examples](quickstart.md#run-a-cis-scan) for complete permutation and SPA + ACAT commands.

!!! tip "Strongly recommended: use SPA with ACAT"

    For score-test ACAT scans, use `--spa --acat`. ACAT directly transforms the variant p-values and is sensitive
    to inaccurate tail probabilities: a poorly calibrated variant can disproportionately affect the gene-level
    result. SPA improves score-tail calibration before aggregation; it does not guarantee that every tail
    calculation succeeds.

ACAT and Beta permutation use different calibration procedures and need not produce the same p-values or
discoveries. Both require appropriate models and valid numerical results. An ACAT p-value accounts for aggregation
within a gene; testing many genes still requires study-level multiple-testing control. See
[Post-process cis results](postprocessing.md).

### Permutation calibration

Permutation calibration uses the maximum absolute score or Wald statistic across the window. Selecting `--spa`
does not replace those permutation statistics with saddlepoint p-values. For ACAT, the returned per-variant
p-values, including successful SPA corrections, are the inputs to the gene-level test.

Beta permutation does not have this same dependence on accurate asymptotic variant p-values. It calibrates the
observed statistic against statistics computed with the same procedure under permutation. Distortions arising
from geometry shared by the observed and permuted tests can therefore be reflected in the permutation reference
distribution. This relies on valid permutations and successful fitting; it does not correct numerical failures
or guarantee the fitted Beta approximation is accurate.

See [Hypothesis testing](../api/hypothesis/variant.md) and
[Gene-level aggregation](../api/hypothesis/gene.md) for the Python interfaces.
