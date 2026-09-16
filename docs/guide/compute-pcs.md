# Compute expression PCs

Compute expression principal components to use as optional mapping covariates. Run PCA separately for each
cell type's pseudobulk matrix, and choose the number of covariates as part of your study design.

This guide uses the tutorial files from the [Quickstart](quickstart.md).
For option defaults, input constraints, and preprocessing details, see the
[Expression PCA command reference](../reference/compute-pcs.md).

## Compute PCs from raw counts

```bash
jaxqtl compute-pcs \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --num-pcs 2 \
  --out tutorial/output/CD4_NC.N100.expr_pcs.tsv
```

The defaults scale counts to the median library size, then apply `log1p`. PCA centers and scales variable genes.
No genotype or covariate file is required.

The output TSV contains `iid`, `ExprPC1`, and `ExprPC2`. A companion
`CD4_NC.N100.expr_pcs.tsv.variance.tsv` records each component's explained-variance proportion.
The log records filtering summaries and these proportions too.

## Add PCs to your mapping covariates

Supply your covariate table so PCA uses the shared samples and writes a combined table:

```bash
jaxqtl compute-pcs \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --covar tutorial/input/donor_features.tsv \
  --num-pcs 2 \
  --out tutorial/output/covariates_with_pcs.tsv
```

Use `tutorial/output/covariates_with_pcs.tsv` as `--covar` in your mapping command.
The original covariate values are copied into this file. They are not regressed out of expression during PCA.

## Select an analysis cohort and gene set

For your own dataset, prepare `analysis_samples.txt` and `pca_genes.txt`, each containing one ID per line
without a header:

```bash
jaxqtl compute-pcs \
  --pheno expression.bed.gz \
  --covar covariates.tsv \
  --keep analysis_samples.txt \
  --gene-list pca_genes.txt \
  --min-indiv-expr-pct 0.1 \
  --min-gene-expr-pct 0.1 \
  --num-pcs 10 \
  --out covariates_with_pcs.tsv
```

This keeps samples expressing more than 10% of input genes, then genes expressed in more than 10% of
retained samples. Library sizes still use all loaded genes, preserving the sequencing-depth adjustment
when you select a smaller PCA gene set.

## Use TMM normalization

To adjust library sizes for expression composition, select TMM:

```bash
jaxqtl compute-pcs \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --normalization tmm \
  --num-pcs 2 \
  --out tutorial/output/expr_pcs_tmm.tsv
```

The default log transform still follows normalization. Supply the full count matrix and select PCA genes
with `--gene-list` when possible: TMM estimates its factors before that selection.

If you must use a count matrix already restricted to a gene subset, provide library sizes from the original
matrix using `--libsize library_sizes.tsv`. See the
[library-size input contract](../reference/compute-pcs.md#normalization-and-transformation) for the table format
and limitations of TMM on restricted input.

## Use expression that is already transformed

For expression that has already been normalized and log-transformed, disable both stages:

```bash
jaxqtl compute-pcs \
  --pheno normalized_log_expression.bed.gz \
  --normalization none \
  --transform none \
  --num-pcs 10 \
  --out expr_pcs.tsv
```

PCA still centers and scales variable genes. If normalization is complete but a log transform is still needed,
use `--normalization none --transform log1p`.

## Plot variance explained

The saved variance table lets you make a scree plot without loading expression again.
With Matplotlib installed, plot the table from the first example:

```python
import matplotlib.pyplot as plt
import polars as pl

variance = pl.read_csv(
    "tutorial/output/CD4_NC.N100.expr_pcs.tsv.variance.tsv",
    separator="\t",
)
fig, ax = plt.subplots()
ax.plot(
    variance["component"].to_list(),
    100 * variance["explained_variance_ratio"].to_numpy(),
    marker="o",
)
ax.set(xlabel="Expression component", ylabel="Variance explained (%)")
fig.tight_layout()
fig.savefig("expression_pcs_scree.png", dpi=150)
```

The proportions refer to transformed, gene-standardized expression and need not sum to 100% when only some
components were requested. Use the saved proportions rather than the variance of the PC columns themselves.
See the [output reference](../reference/compute-pcs.md#outputs) for their definition.
