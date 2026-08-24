# Cis and nominal mapping

`jaxqtl` supports cis-eQTL mapping by scanning variants within a window around each gene (or other molecular feature).

## Modes

`jaxqtl` exposes two cis-related scanning modes:

- **cis**: per gene, compute variant-level statistics in the cis window and return the lead SNP plus a gene-level
  adjusted p-value (e.g. via permutation/Beta approximation or ACAT).
- **nominal**: per gene, return statistics for all variants in the cis window.

The Python mapper consumes a `ReadyDataState` whose genotype, expression, covariates, and offsets have already been
aligned on IID. The CLI constructs this state internally.

!!! note "The mapping state is an advanced interface"

    `ReadyDataState` is not currently re-exported from `jaxqtl.map`. Import it from `jaxqtl.map.data` when using the
    mapping function directly.

::: jaxqtl.map.data.ReadyDataState
    options:
        members:
            - from_data
            - iter_cis

---

::: jaxqtl.map.map_cis
