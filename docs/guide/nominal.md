# Nominal mapping

Nominal mapping reports every variant association within each phenotype's cis window. Use it when downstream analysis
needs a complete local summary-statistics table rather than one calibrated lead association per gene.

```bash
jaxqtl nominal \
  --bfile tutorial/input/chr22_N100 \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --covar tutorial/input/donor_features.tsv \
  --gene-list tutorial/input/genelist_10 \
  --model nb \
  --test wald \
  --set-offset-from-libsize \
  --normalize-covar \
  --out tutorial/output/nominal
```

This writes `tutorial/output/nominal.nominal.wald.parquet.gz`.

The `--window` value is the number of bases added on each side of the phenotype interval and defaults to 500,000.
Unlike `cis`, nominal mode does not run Beta-permutation calibration or ACAT.

See [Nominal output](../reference/outputs.md#nominal-output) for the output columns.
