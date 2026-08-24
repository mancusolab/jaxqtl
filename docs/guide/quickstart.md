# Quickstart

This example runs a small cis-eQTL scan over ten genes from the bundled chr22 tutorial data.

## Get the example data

Clone the repository if you installed jaxQTL without its tutorial files:

```bash
git clone https://github.com/mancusolab/jaxqtl.git
cd jaxqtl
```

## Run a cis scan

```bash
jaxqtl cis \
  --bfile tutorial/input/chr22_N100 \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --covar tutorial/input/donor_features.tsv \
  --gene-list tutorial/input/genelist_10 \
  --model nb \
  --test score \
  --set-offset-from-libsize \
  --normalize-covar \
  --nperm 1000 \
  --out tutorial/output/quickstart
```

!!! warning "Count models require the correct offset"

    `--set-offset-from-libsize` computes `log(library size)` from every phenotype still present in the input file.
    If the phenotype file has already been restricted, supply the precomputed log offset with `--offset` instead.
    See [Offsets](offsets.md) before analyzing production data.

The command writes `tutorial/output/quickstart.cis.score.perm.parquet.gz`. Each row reports the lead variant and a
gene-level adjusted p-value. See [Cis output](../reference/outputs.md#cis-output) for the complete schema and validity
fields.

## Next steps

- Use [Cis mapping](cis.md) to choose between permutation calibration and ACAT.
- Use [Nominal mapping](nominal.md) to retain every association in each cis window.
- Review [Input formats](../reference/inputs.md) before substituting your own data.
