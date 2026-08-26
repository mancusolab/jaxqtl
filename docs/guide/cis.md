# Cis mapping

Cis mapping tests variants within a window around each molecular phenotype and reports one selected association per
phenotype. The output includes the lead variant, its nominal association, and a gene-level adjusted p-value.

The `--window` value defaults to 500,000 bases. By default, the interval extends from `TSS - window` through
`TES + window`. Add `--tss-centered` to instead use `TSS - window` through `TSS + window`.

## Permutation calibration

The default procedure permutes the phenotype, records an extreme statistic across the cis window, and fits a Beta
approximation to the resulting permutation p-values.

```bash
jaxqtl cis \
  --bfile tutorial/input/chr22_N100 \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --covar tutorial/input/donor_features.tsv \
  --gene-list tutorial/input/genelist_10 \
  --model nb \
  --test score \
  --set-offset-from-libsize \
  --nperm 1000 \
  --out tutorial/output/cis_perm
```

## SPA and ACAT

Use SPA for score-test tail calibration and ACAT for gene-level aggregation without permutations:

```bash
jaxqtl cis \
  --bfile tutorial/input/chr22_N100 \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --covar tutorial/input/donor_features.tsv \
  --gene-list tutorial/input/genelist_10 \
  --model nb \
  --test score \
  --spa \
  --acat \
  --set-offset-from-libsize \
  --out tutorial/output/cis_acat
```

!!! note "Cis mode retains some failed tests"

    If every SNP-level p-value for a tested gene is non-finite, jaxQTL writes one row with `result_valid = false` and
    `failure_reason = "no_finite_pvalues"`. Association and lead-variant fields are null because no lead exists.

Genes with no variants in the requested window or no phenotype variance are skipped. See
[Cis output](../reference/outputs.md#cis-output) for the complete result contract.
