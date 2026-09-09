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

Choose permutation calibration or SPA + ACAT. Both report a lead variant and a gene-level p-value; testing does
not require permutations.


| Approach | Why choose it | Main consideration |
| --- | --- | --- |
| SPA + ACAT | Fast gene-level testing without permutations | Sensitive to variant p-value calibration; SPA strongly recommended |
| Beta permutation | Calibrate statistics against a permutation reference | More computation; requires valid permutations and successful calibration |

!!! warning "Compute offsets from the full gene matrix"

    `--set-offset-from-libsize` computes `log(library size)` from every phenotype still present in the input file.
    If the phenotype file has already been restricted, supply the precomputed log offset with `--offset` instead.
    See [Offsets](offsets.md) before analyzing production data.

### Permutation calibration

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

The command writes `tutorial/output/quickstart.cis.score.perm.parquet.gz`. Each row reports the lead variant and a
gene-level adjusted p-value. See [Cis output](../reference/outputs.md#cis-output) for the complete schema and validity
fields.

### Faster scans with SPA and ACAT

SPA + ACAT is typically substantially faster than permutation scans because it fits each gene's null model once
and avoids repeatedly fitting and testing shuffled phenotypes. The speed difference depends on the data and the
number of permutations used for comparison.

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

The command writes `tutorial/output/quickstart_spa_acat.cis.score.spa.acat.parquet.gz`. No permutations are run,
and `--nperm` does not control this procedure. The offset requirements above apply to both examples.

!!! tip "Strongly recommended: use SPA with ACAT"

    ACAT is sensitive to inaccurate variant p-values, so use `--spa` with `--acat` for score tests. SPA improves
    tail calibration but can fall back to the normal approximation when its numerical correction fails.

ACAT and Beta permutation use different calibration procedures and need not produce the same p-values or
discoveries. Check convergence and finite adjusted p-values, and apply multiple-testing correction across genes
with either method. See [Tests and gene-level calibration](tests.md#tail-and-gene-level-calibration) for the tradeoffs
and [Post-process cis results](postprocessing.md) for output checks.

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

A completed command can retain failed results. Before identifying discoveries, follow
[Post-process cis results](postprocessing.md) to filter validity and convergence flags, exclude nonfinite
adjusted p-values, and control FDR across genes. `pvalue_adj` adjusts within a gene's cis window; it is not an
across-gene FDR value.

## Next steps

- Use the [single-cell cis-eQTL workflow](single-cell-cis.md) to prepare and run your own cell-type-specific data.
- Use [Cis mapping](cis.md) to choose between permutation calibration and ACAT; SPA is strongly recommended with ACAT.
- Use [Nominal mapping](nominal.md) to retain every association in each cis window.
- Use [Post-process cis results](postprocessing.md) to filter failures and apply a study-level FDR procedure.
- Review [Data preparation](single-cell-cis.md) before substituting your own data.
