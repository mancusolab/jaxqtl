# Expression data

`jaxqtl` maps genetic variants against molecular phenotypes (e.g. gene expression). Phenotype data is handled as:

- a sample-by-feature matrix (individuals x genes)
- a feature metadata table (chromosome, start, end, feature ID)
- optional library sizes used to construct offsets for count-based models

See [Phenotype data](../../guide/phenotypes.md) for file formats and coordinates, and
[Offsets](../../guide/offsets.md) for the unfiltered-input requirement.

## Expression container

::: jaxqtl.io.ExpressionData
    options:
        members:
            - from_bedfile
            - to_jax
            - offset_from_libsize
            - filter_genes_by_percentage
            - filter_individuals_by_percentage
            - compute_pcs
