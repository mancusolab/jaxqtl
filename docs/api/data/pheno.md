# Expression data

`jaxqtl` maps genetic variants against molecular phenotypes (e.g. gene expression). Phenotype data is handled as:

- a sample-by-feature matrix (individuals x genes)
- a feature metadata table (chromosome, start, end, feature ID)
- optional library sizes used to construct offsets for count-based models

## Input formats

`jaxqtl` expects BED-like expression inputs containing at least:

1. `chrom`
2. `start`
3. `end`
4. `phenotype_id` (or a supported alias)
5. one column per sample IID

Coordinates are passed through to define mapping windows. Use the same chromosome naming and coordinate convention as
the genotype source.

## Offsets

Count models (Poisson/NB) often use an offset (e.g. log library size). Offsets can be:

- provided explicitly, matched by IID
- computed from the loaded phenotype matrix

See [Offsets](../../guide/offsets.md) for the unfiltered-input requirement.

## API

::: jaxqtl.io.ExpressionData
    options:
        members:
            - from_bedfile
            - to_jax
            - offset_from_libsize
            - filter_genes_by_percentage
            - filter_individuals_by_percentage
            - compute_pcs
