# Gene-level aggregation

Aggregations return one gene-level p-value and diagnostics. ACAT combines real variants' p-values;
BetaPermutation evaluates the selected lead statistic against a reference built from permutation maxima.

**SPA is strongly recommended when ACAT aggregates score-test p-values.** Use `SpaTest` with `ACAT`
in Python, or `--spa --acat` in the CLI. ACAT's sensitivity to inaccurate tail probabilities makes variant-level
calibration important. See [Tests and gene-level calibration](../../guide/tests.md#tail-and-gene-level-calibration)
for the distinction from Beta permutation, which calibrates statistics against their permutation distribution.

??? abstract "`jaxqtl.hypothesis.AbstractAggregateTest`"

    ::: jaxqtl.hypothesis.AbstractAggregateTest
        options:
            show_bases: true
            members:
                - statistic
                - init
                - update
                - finalize

## Aggregation methods

::: jaxqtl.hypothesis.BetaPermutation
    options:
        show_bases: true
        members:
            - __init__
            - statistic
            - init
            - update
            - finalize
            - adjust

---

::: jaxqtl.hypothesis.ACAT
    options:
        show_bases: true
        members:
            - __init__
            - statistic
            - init
            - update
            - finalize

## Reduce blocks and finalize

`AbstractAggregateTest[ReductionStateT, ReferenceT, Aux]` defines a shared numerical lifecycle:

```python
state = method.init(dtype, num_variants=number_of_real_variants)
values = method.statistic(block_result)
state = method.update(state, values, valid_mask)  # Repeat for each block.
pvalue, diagnostics = method.finalize(state, reference)
```

`statistic` selects p-values for ACAT or z statistics for Beta permutation inside the compiled kernel.
This lets permutation kernels discard unused SPA tail calculations. `valid_mask` excludes padding.
State shapes do not depend on window width; ACAT weights use the full count of real variants.

| Method | Reduction state | Reference | Finalization |
| --- | --- | --- | --- |
| ACAT | `CauchyState`: weighted sum, endpoint flags, and weight | `None` | Convert the complete Cauchy statistic to a p-value |
| Beta permutation | Scalar array: maximum absolute z statistic | `PermutationReference`: permutation maxima and residual degrees of freedom | Fit calibration and evaluate an observed statistic |

Permutation reductions produce one maximum per shuffle and are not individually finalized. Cis orchestration
selects the lead once, passes its index to the formatter, and calls `finalize(lead_z, reference)`.
Selection uses the nominal or SPA p-value, which need not identify the largest absolute z statistic.

## Result type

`PermutationResult` is the public type alias for the `(pvalue, auxiliary_diagnostics)` tuple returned by aggregation
methods. Both blocked scans and `jaxqtl.map.cis.map_cis_single` return one scalar gene-level
p-value for either aggregation. `map_cis_single` orchestrates compiled full-window kernels and lead selection
on the host; its wrapper is not JIT-transformable.

Beta permutation's `finalize` accepts one scalar lead statistic. Applying an existing calibration to
additional SNPs is a separate operation:

```python
method = BetaPermutation()
result, (gene_pvalue, calibration) = map_cis_single(
    X, G, y, offset, snp_test=test, gene_test=method, key=key
)
snp_adjusted_pvalues = method.adjust(result.z, calibration)
```

These SNP values use the gene's permutation-maximum reference for within-gene multiple testing adjustment.
This operation does not provide marginal p-value calibration like SPA and does not refit the calibration.

`BetaCalibration` names the auxiliary fields `beta_params`, `reference_estimate`, and `reference_converged`.
The diagnostics distinguish the fitted Beta parameters from convergence of the reference-distribution estimate.

## Execution and extension

`AssociationScan` owns fitting, block transfers, and permutation batching. Cis orchestration owns lead
selection and finalization; aggregators own statistical calculations.

Custom observed-only aggregators implement `statistic`, `init`, `update`, `finalize`, and `name`.
Their preferred `block_size` selects blocked execution; `None` selects a full window.
A different resampling workflow requires extending the executor. Cis output currently supports ACAT and
BetaPermutation; adding another method also requires extending the formatter.

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

::: jaxqtl.hypothesis.CauchyState
    options:
        members: false

::: jaxqtl.hypothesis.PermutationReference
    options:
        members: false

## Beta approximation for permutation p-values

For cis mapping, `jaxqtl` can fit a Beta approximation to the distribution of permutation p-values:

::: jaxqtl.infer.infer_beta_params

---

::: jaxqtl.infer.BetaParams
    options:
        members: false
