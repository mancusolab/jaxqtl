# Gene-level aggregation

For cis mapping, `jaxqtl` supports gene-level calibration and aggregation over the set of variants tested in a cis
window.

Aggregations combine per-variant results and, for permutations, recompute statistics under shuffled outcomes.
They return adjusted p-values with method-specific auxiliary diagnostics. ACAT returns one gene-level p-value;
Beta permutation returns one adjusted value per variant, from which cis mapping selects the lead variant.

**SPA is strongly recommended when ACAT aggregates score-test p-values.** Use `SpaTest` with `ACAT`
in Python, or `--spa --acat` in the CLI. ACAT's sensitivity to inaccurate tail probabilities makes variant-level
calibration important. See [Tests and gene-level calibration](../../guide/tests.md#tail-and-gene-level-calibration)
for the distinction from Beta permutation, which calibrates statistics against their permutation distribution.

??? abstract "`jaxqtl.hypothesis.AbstractAggregateTest`"

    ::: jaxqtl.hypothesis.AbstractAggregateTest
        options:
            show_bases: true
            members:
                - aggregate
                - scan
                - __call__

## Aggregation methods

::: jaxqtl.hypothesis.BetaPermutation
    options:
        show_bases: true
        members:
            - __init__

---

::: jaxqtl.hypothesis.ACAT
    options:
        show_bases: true
        members:
            - __init__

## Result type

`PermutationResult` is the public type alias for the `(pvalue, auxiliary_diagnostics)` tuple returned by aggregation
methods. The p-value component can be scalar or variantwise, depending on the method.

`BetaCalibration` names the auxiliary fields `beta_params`, `reference_estimate`, and `reference_converged`.
The diagnostics distinguish the fitted Beta parameters from convergence of the reference-distribution estimate.

For cis execution, each aggregation class implements `scan(execution, X, G, y, offset, key)`. ACAT owns its
masked Cauchy contributions, accumulator update, and final conversion. Beta permutation owns calibration and
application to observed statistics; the executor batches initialization and fixed-block evaluation through the
hypothesis test's `init`/`test` interface. `aggregate` and `__call__` provide the full-array transformable API.

The executor lives in `jaxqtl.map`; aggregation classes use the `ScanExecution` protocol defined alongside their
abstract interface. Host packing and transfers are execution details and do not enter the statistical methods.

Aggregation classes declare `adjustment_method`, `has_calibration`, and a preferred `block_size`; `None` selects
full-window execution. Output formatting uses this metadata explicitly. A custom aggregation must implement
`scan` and the metadata contract as well as `aggregate` and `name`.

## Calibration and failure behavior

Beta permutation records the maximum absolute score or Wald statistic over the entire cis window for each
permutation. SPA uses the underlying score statistic in this procedure; its tail-corrected p-values enter ACAT.
The phenotype and a vector offset are shuffled together, while covariates and genotypes remain fixed.

ACAT weights each real variant by the inverse of the number of variants in the window. Padding has zero
contribution. Nonfinite input p-values propagate through the aggregate; inputs containing both exact zero and
exact one trigger an error. Values close to one can dominate the negative side of the Cauchy sum.

A finite lead-variant p-value does not guarantee a finite adjusted p-value or successful Beta calibration. Check
the convergence fields and the adjusted p-value as described in [Troubleshooting](../../guide/troubleshooting.md).

::: jaxqtl.hypothesis.BetaCalibration
    options:
        members: false

## Beta approximation for permutation p-values

For cis mapping, `jaxqtl` can fit a Beta approximation to the distribution of permutation p-values:

::: jaxqtl.infer.infer_beta_params

---

::: jaxqtl.infer.BetaParams
    options:
        members: false
