# Gene-level aggregation

For cis mapping, `jaxqtl` supports gene-level calibration and aggregation over the set of variants tested in a cis
window.

Aggregations reduce per-variant results across a cis window and return a p-value with method-specific diagnostics.
Blocked cis scans return one adjusted value per gene. Beta permutation evaluates the selected lead's statistic
against a reference built from permutation maxima; ACAT combines all real variants' p-values.

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
Unused SPA tail calculations can therefore be eliminated during permutation testing. `valid_mask` excludes padded variants. Each state has a fixed shape independent of window width. The whole-window
variant count determines ACAT weights, including when the last block is partial.

| Method | Reduction state | Reference | Finalization |
| --- | --- | --- | --- |
| ACAT | `CauchyState`: weighted sum, endpoint flags, and weight | `None` | Convert the complete Cauchy statistic to a p-value |
| Beta permutation | Scalar array: maximum absolute z statistic | `PermutationReference`: permutation maxima and residual degrees of freedom | Fit calibration and evaluate an observed statistic |

The mapper accumulates one scalar maximum per permutation across genotype blocks. These
reference reductions are not individually finalized. Cis orchestration selects the lead once and passes
the same index to the output formatter. It calls `finalize(lead_z, reference)` after collecting the permutation maxima. Selecting the observed statistic by nominal or SPA p-value preserves
lead selection when the best p-value does not correspond to the maximum absolute z statistic.
There is no separate observed-maximum accumulator.

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

For cis execution, `AssociationScan` in `jaxqtl.map` runs observed scans and, for Beta permutation,
additional permutation scans. Cis orchestration owns lead selection and invokes finalization;
the scan executor owns
block scheduling, permutation batching, compiled calls, and host transfers through the hypothesis test's
`init`/`test` interface. Aggregation classes contain the statistical calculations: ACAT contributions, accumulator
updates, and final conversion; maximum-statistic reduction, Beta calibration, and adjustment of observed statistics. Both execution paths
reuse the same numerical methods; aggregation classes do not fit hypothesis tests or schedule scans.

Aggregation classes declare a preferred `block_size`; `None` selects full-window execution.
The cis output formatter includes calibration columns for BetaPermutation.
A custom observed-only aggregation implements `statistic`, `init`, `update`, `finalize`, and `name`.
It can use full-window or blocked execution without changes to `AssociationScan`. A different resampling
workflow requires extending the executor.
The cis output formatter assigns `"ACAT"` and `"BETA"` labels from the concrete aggregation type. Supporting a
different aggregation in cis output requires extending the formatter; unsupported types raise `TypeError`.

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
