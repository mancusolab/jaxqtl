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

For a single-cell count-model analysis, aggregate cells by sum within each donor and annotated cell type. The result
is one sample-level count for each donor, cell type, and gene. Summation preserves the exposure represented by the
library-size offset; averaging cells does not.

Write a separate phenotype file for each cell type. jaxQTL does not read a cell-type column or split a matrix by cell
type. See the [single-cell cis-eQTL workflow](single-cell-cis.md) for the complete preparation and mapping process.

Fractional abundance estimates produced by a quantifier are valid inputs. jaxQTL does not require every observed
phenotype value to be an integer.

!!! note "Pseudobulking is specific to single-cell count workflows"

    Gaussian analyses and molecular phenotypes that are already measured at the sample level do not require this
    preprocessing step.

## Expression filters

`--min-gene-expr-pct` removes genes that are nonzero in too few individuals. `--min-indiv-expr-pct` removes
individuals that are nonzero for too few genes. Values are proportions between 0 and 1.

!!! warning "Filtering can change an on-the-fly library size"

    `--set-offset-from-libsize` can only use genes present in the phenotype file. If the file was filtered before
    jaxQTL reads it, use a precomputed offset from the unfiltered count matrix.

!!! warning "Every retained donor needs positive total abundance"

    Omit donor–cell-type combinations with no retained cells or zero total abundance. Their log-library-size offset
    is not finite.

See [Offsets](offsets.md) for the count-model exposure contract.
