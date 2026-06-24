# Cis Mapping

Cis mapping tests variants within a window around each gene (or other molecular feature).

## Typical workflow

1. Load genotypes from one of:
   - `--bfile` (PLINK1 BED/BIM/FAM prefix)
   - `--pfile` (PLINK2 PGEN/PVAR/PSAM prefix)
   - `--vcf` (indexed VCF/BCF)
   - `--bgen` (BGEN)
2. Load expression/phenotypes and covariates.
3. Choose a model (`gaussian`, `poisson`, `nb`) and a test (`score`, `wald`).
4. Run `jaxqtl cis` to obtain lead variants and optionally gene-level calibrated p-values.

See `API → molQTL Mapping → Cis Mapping` for the API entrypoints.
