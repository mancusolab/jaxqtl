# File Formats

`jaxqtl` uses a small set of standard file formats for inputs and outputs.

## Inputs

### Genotypes

- PLINK bed/bim/fam triplets via a prefix (`--bfile` / `--geno`)
- VCF files (`--vcf`)

### Phenotypes (expression)

BED-like tables containing feature metadata columns (chrom/start/end/ID) followed by one column per sample IID.

### Covariates and offsets

TSV/CSV-like tables with an `iid` column and one or more covariate/offset columns.

## Outputs

Mapping outputs are written as tabular files (e.g. TSV/Parquet) containing:

- variant metadata
- per-variant summary statistics (effect sizes, standard errors, p-values)
- optional gene-level adjusted p-values for cis scans
