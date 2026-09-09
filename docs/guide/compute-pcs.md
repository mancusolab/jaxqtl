# Compute expression PCs

Expression PCs are optional covariates; choose whether to include them as part of the study design.

`jaxqtl compute-pcs` estimates expression principal components and appends them to an existing covariate table.

```bash
jaxqtl compute-pcs \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --covar tutorial/input/donor_features.tsv \
  --num-pcs 2 \
  --transform log1p \
  --out tutorial/output/CD4_NC.N100.covar_with_expr_pcs.tsv
```

The output contains the original covariates followed by `ExprPC0`, `ExprPC1`, and subsequent requested components.
Pass this table to `--covar` in a mapping command.

Each row represents one individual, with exactly `--num-pcs` component columns. Components are unit-norm sample
directions ordered by decreasing explained variance within the estimated subspace. The algorithm estimates that
subspace iteratively, then uses a small projected SVD to resolve its principal directions.

For a scree plot, evaluate explained variance on the same expression matrix used for fitting: apply the selected
transform, then center and scale each gene to unit variance across samples. Variance explained in unscaled
expression need not decrease in component order.

For cell-type-specific analyses, compute expression PCs separately from each cell type's pseudobulk matrix.

Expression PCA uses a probabilistic algorithm. Reusing `--seed` with the same inputs makes initialization
reproducible, although floating-point results can vary across JAX backends.

`--num-pcs` must be positive and cannot exceed the smaller of the sample and phenotype counts. The optional `log1p`
transform is available; `tmm` is currently not implemented.
