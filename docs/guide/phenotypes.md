# Phenotype data

jaxQTL reads BED-like, tab-delimited expression matrices. The first four columns contain feature metadata; remaining
columns contain one sample per column:

```text
#Chr  start  end  Geneid  sample_1  sample_2  ...
1     29553  29554 ENSG... 14        9         ...
```

For gene expression, calculate the strand-aware transcription start site (TSS) and encode it as a one-base,
zero-based BED interval: `start = TSS - 1` and `end = TSS`, where `TSS` is the one-based genomic position.
Chromosome labels must match the genotype source.

Accepted aliases are case-insensitive:

1. Chromosome: `chrom`, `#chrom`, `chr`, or `#chr`.
2. Start: `start`.
3. End: `end`.
4. Phenotype ID: `pheno_id`, `pheno`, `gene_id`, `geneid`, or `gene`.

Inputs may end in `.bed`, `.bed.gz`, `.parquet`, or `.parquet.gz`.

## Single-cell pseudobulk counts

For single-cell count models, supply one summed donor-level pseudobulk matrix per cell type. See the
[single-cell workflow](single-cell-cis.md) for preparation; jaxQTL does not split a matrix by cell type.

Fractional abundance estimates produced by a quantifier are valid inputs. jaxQTL does not require every observed
phenotype value to be an integer.

Gaussian analyses and molecular phenotypes that are already measured at the sample level do not require this
preprocessing step.

## Expression filters

`--min-gene-expr-pct` removes genes that are nonzero in too few individuals. `--min-indiv-expr-pct` removes
individuals that are nonzero for too few genes. Values are proportions between 0 and 1.

!!! warning "Compute offsets before restricting the phenotype file"

    `--set-offset-from-libsize` can only use genes present in the phenotype file. If the file was filtered before
    jaxQTL reads it, use a precomputed offset from the unfiltered count matrix.

!!! warning "Exclude donors with zero total abundance"

    Omit donor–cell-type combinations with no retained cells or zero total abundance. Their log-library-size offset
    is not finite.

See [Offsets](offsets.md) for the count-model exposure contract.
