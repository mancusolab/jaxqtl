# File Formats

`jaxqtl` uses a small set of standard file formats for inputs and outputs.

## Inputs

### Genotypes

- PLINK1 BED/BIM/FAM triplets via a prefix (`--bfile`)
- `--geno` is deprecated; use `--bfile` for PLINK1 BED/BIM/FAM prefixes.
- VCF (`--vcf`) is currently unsupported/experimental and is not a production genotype input.

### Phenotypes (expression)

BED-like tables containing feature metadata columns (chrom/start/end/ID) followed by one column per sample IID.

### Covariates and offsets

TSV/CSV-like tables with an `iid` column and one or more covariate/offset columns.

## Outputs

Mapping outputs are written as tabular files (e.g. TSV/Parquet) containing:

- variant metadata
- per-variant summary statistics (effect sizes, standard errors, p-values)
- optional gene-level adjusted p-values for cis scans
