# Gene-level aggregation

For cis mapping, `jaxqtl` supports gene-level calibration and aggregation over the set of variants tested in a cis
window.

Aggregation procedures take:

- per-variant test results for a gene window
- a mapping/test callable to recompute statistics under resampling/permutation

and return calibrated p-values with method-specific auxiliary diagnostics. ACAT returns one gene-level p-value;
Beta permutation returns one adjusted value per variant, from which cis mapping selects the lead variant.

??? abstract "`jaxqtl.hypothesis.AbstractAggregateTest`"

    ::: jaxqtl.hypothesis.AbstractAggregateTest
        options:
            show_bases: true
            members:
                - aggregate
                - __call__

## Implementations

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
