# File Formats

`jaxqtl` uses a small set of standard file formats for inputs and outputs.

## Inputs

### Genotypes

- Production CLI genotype support in this implementation is PLINK1 BED/BIM/FAM through `--bfile`.
- `--geno` is deprecated and now raises an error; use `--bfile` for PLINK1 BED/BIM/FAM prefixes.
- `--vcf` remains unsupported in production CLI behavior for this migration.
- PLINK2, VCF/BCF, BGEN, dosage, sparse, and haplotype inputs are future extension paths behind later design work.

### Phenotypes (expression)

BED-like tables containing feature metadata columns (chrom/start/end/ID) followed by one column per sample IID.

### Covariates and offsets

TSV/CSV-like tables with an `iid` column and one or more covariate/offset columns.

## Outputs

Mapping outputs are written as tabular files (e.g. TSV/Parquet) containing:

- variant metadata
- per-variant summary statistics (effect sizes, standard errors, p-values)
- optional gene-level adjusted p-values for cis scans
