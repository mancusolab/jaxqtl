# Quickstart

These examples run small cis-eQTL scans over ten genes from the bundled chr22 tutorial data. The `CD4_NC` phenotype
file is already a donor-by-gene pseudobulk matrix for one cell type. To prepare this input from cell-level data, start
with the [single-cell cis-eQTL workflow](single-cell-cis.md).

## Get the example data

Clone the repository if you installed jaxQTL without its tutorial files:

```bash
git clone https://github.com/mancusolab/jaxqtl.git
cd jaxqtl
```

## Run a cis scan

Both approaches report a lead variant and a gene-level p-value:

| Approach | Why choose it | Main consideration |
| --- | --- | --- |
| SPA + ACAT | Fast gene-level testing without permutations | Sensitive to variant p-value calibration; SPA strongly recommended |
| Beta permutation | Calibrate statistics against a permutation reference | More computation; requires valid permutations and successful calibration |


### Permutation calibration

Permutation testing fits a Beta approximation to the permutation-derived null distribution to estimate a gene-level p-value.

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

The command writes `tutorial/output/quickstart.cis.score.perm.parquet.gz`.

!!! warning "Compute offsets from the full gene matrix"

    `--set-offset-from-libsize` computes `log(library size)` from every phenotype still present in the input file.
    If the phenotype file has already been restricted, supply the precomputed log offset with `--offset` instead.
    See [Offsets](offsets.md) before analyzing production data.

### Faster scans with SPA and ACAT

SPA + ACAT avoids permutation refits and is typically much faster. The speedup depends on the data and
permutation count.

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
  --spa \
  --acat \
  --out tutorial/output/quickstart_spa_acat
```

The command writes `tutorial/output/quickstart_spa_acat.cis.score.spa.acat.parquet.gz`.

!!! warning "Use SPA with score-test ACAT"

    ACAT can amplify inaccurate variant tail p-values into misleading gene-level results. We strongly recommend
    `--spa --acat` for score tests. SPA can still fall back to the normal approximation.

The methods can yield different p-values and discoveries. See [Calibration tradeoffs](tests.md#tail-and-gene-level-calibration);
both require result checks and FDR correction across genes.

## Inspect the result

Read the file produced by the command you chose. This small tutorial table is loaded into memory for inspection:

```python
import polars as pl

path = "tutorial/output/quickstart_spa_acat.cis.score.spa.acat.parquet.gz"
# For Beta permutation, use "tutorial/output/quickstart.cis.score.perm.parquet.gz".
results = pl.read_parquet(path)
checks = ["result_valid", "model_converged"]
if "perm_converged" in results.columns:
    checks.append("perm_converged")
print(results.select("phenotype_id", "pvalue_adj", *checks))
```

`pvalue_adj` adjusts within a cis window; it is **not an across-gene FDR value**.
Follow [Post-process cis results](postprocessing.md) to filter failed/nonfinite results and control FDR.

## Next steps

- [Single-cell workflow](single-cell-cis.md): prepare and analyze your own data.
- [Cis mapping](cis.md): choose regions, filters, and fitting settings.
- [Nominal mapping](nominal.md): retain every variant association.
