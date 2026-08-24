# Compute expression PCs

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

!!! note "The seed controls randomized initialization"

    Expression PCA uses a probabilistic algorithm. Reusing `--seed` with the same inputs makes initialization
    reproducible, although floating-point results can vary across JAX backends.

`--num-pcs` must be positive and cannot exceed the smaller of the sample and phenotype counts. The optional `log1p`
transform is available; `tmm` is currently not implemented.
