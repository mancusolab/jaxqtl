# Trans mapping

Trans mapping tests genotype blocks against every retained phenotype. jaxQTL streams blocks from the genotype source
so the entire variant matrix does not need to reside in memory.

```bash
jaxqtl trans \
  --bfile tutorial/input/chr22_N100 \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --covar tutorial/input/donor_features.tsv \
  --model nb \
  --test score \
  --set-offset-from-libsize \
  --out tutorial/output/trans
```

The current CLI reads 2,500 variants per block. It writes variant metadata separately from the phenotype-by-variant
summary statistics so metadata are not repeated for every phenotype.

!!! note "Constant phenotypes are excluded"

    Phenotypes with zero or NaN variance are removed before the scan. If none remain, jaxQTL writes no result blocks
    and records a warning.

See [Trans output](../reference/outputs.md#trans-output) for filenames and schemas.
