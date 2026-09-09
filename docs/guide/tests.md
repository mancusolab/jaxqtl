# Tests and gene-level calibration

Choose a [response model](model.md), a variant-level test, and a gene-level calibration method.

## Variant-level tests

The **score test** is the cis default: it fits one covariate-only null model per phenotype and reuses it across
variants. **Wald tests** fit each variant's effect and are the usual choice for nominal scans requiring full-model
effect estimates.

!!! warning "Use robust standard errors only with Wald tests"

    Score and SPA inference do not support Huber–White sandwich errors. Use `--test wald --robust-se` when
    requesting them; this option does not make score or SPA inference robust to model misspecification.

## Tail and gene-level calibration

`--spa` attempts a saddlepoint correction for score statistics with absolute normal z-score above 1.96 that pass
the CGF's support checks. Smaller statistics use the normal tail. A failed root solve or invalid correction also
falls back to the normal approximation.

There is no separate SPA-success flag: `model_converged` describes the null fit, not the tail calculation.

### Faster cis scans with SPA and ACAT

`--spa --acat` combines variant p-values into one gene-level p-value without permutations. It is typically much
faster than Beta permutation because it avoids permutation refits; the speedup depends on the data and permutation
count. See the [Quickstart](quickstart.md#run-a-cis-scan) for both commands.

!!! warning "Use SPA with score-test ACAT"

    ACAT can amplify inaccurate variant tail p-values into misleading gene-level results. We strongly recommend
    `--spa --acat` for score tests. SPA can still fall back to the normal approximation.

### Permutation calibration

Without `--acat`, cis scans use a Beta approximation fitted to permutation maxima. Each shuffle contributes the
maximum absolute score or Wald statistic across the window. `--spa` does not replace these statistics with
saddlepoint p-values.

Beta permutation does not share ACAT's direct sensitivity to asymptotic variant p-values: it compares statistics
computed by the same procedure in the observed and permuted data. Geometry shared across those tests can therefore
be reflected in the reference distribution. This requires valid permutations and successful fitting; it cannot
repair numerical failures or guarantee an accurate Beta approximation.

The methods can yield different discoveries. Both need [result checks and FDR correction across genes](postprocessing.md).

For Python interfaces, see [Variant-level tests](../api/hypothesis/variant.md) and
[Gene-level aggregation](../api/hypothesis/gene.md).
