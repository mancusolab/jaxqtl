# File Formats

`jaxqtl` uses a small set of standard file formats for inputs and outputs.

## Inputs

### Genotypes

- Production CLI genotype support in this implementation is:
  - `--bfile` (PLINK1 BED/BIM/FAM prefix)
  - `--pfile` (PLINK2 PGEN/PVAR/PSAM prefix)
  - `--vcf` (indexed VCF/BCF)
  - `--bgen` (BGEN)
- `--geno` is deprecated and now raises an error; use one of the four supported genotype flags instead.

### Phenotypes (expression)

BED-like tables containing feature metadata columns (chrom/start/end/ID) followed by one column per sample IID.

### Covariates and offsets

TSV/CSV-like tables with an `iid` column and one or more covariate/offset columns.

## Outputs

Mapping outputs are written as tabular files (e.g. TSV/Parquet) containing:

- variant metadata
- per-variant summary statistics (effect sizes, standard errors, p-values)
- optional gene-level adjusted p-values for cis scans

Cis output retains tested genes that produce no finite SNP-level p-values. Such rows have
`result_valid = false`, `failure_reason = "no_finite_pvalues"`, and null lead-variant and association fields.
