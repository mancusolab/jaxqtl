# Models, tests, and calibration

jaxQTL separates the response model, variant-level test, and gene-level calibration method. These choices answer
different questions and should be selected independently.

## Response models

- `gaussian` fits a linear model to a continuous phenotype.
- `poisson` models counts with variance equal to the mean.
- `nb` uses the NB2 variance `mu + alpha * mu^2` for overdispersed counts.

## Variant-level tests

The score test fits one covariate-only model per phenotype and tests each genotype against that null fit. It is the
default for cis scans because the null-model work is reused across variants.

The Wald test fits the tested variant coefficient and reports its estimate, standard error, and Wald statistic. It is
the usual choice for nominal results when effect estimates for every variant are required.

!!! warning "Robust standard errors apply only to Wald tests"

    `--robust-se` selects Huber–White sandwich standard errors for Wald inference. It does not make the implemented
    score statistic or saddlepoint approximation misspecification-robust.

## Tail and gene-level calibration

`--spa` replaces the normal-tail p-value for a score statistic with a saddlepoint approximation. It is intended for
settings where asymptotic normal tails can be inaccurate, such as low minor-allele counts.

For `jaxqtl cis`, the default gene-level procedure uses permutations and a fitted Beta approximation. `--acat`
instead combines the variant p-values with the aggregated Cauchy association test.

See [Hypothesis testing](../api/hypothesis/variant.md) and
[Gene-level aggregation](../api/hypothesis/gene.md) for the Python interfaces.
