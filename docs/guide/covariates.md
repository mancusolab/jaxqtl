# Covariates and sample alignment

Covariate and offset tables are tab-delimited and must contain exactly one case-insensitive `iid` or `#iid` column.
An optional `fid` column is ignored. jaxQTL aligns every input on the intersection of sample IDs and preserves the
genotype source order.

IID values must use the same spelling in every input. Samples missing from any required input are excluded by the
intersection, so compare the IDs before mapping and confirm the resulting sample count.

```text
iid       age  sex  genotype_pc1
sample_1  42   F    -0.013
sample_2  57   M     0.021
```

Use `--covar-name` to retain selected covariates or `--rm-covar` to remove selected covariates. The options are
mutually exclusive. `--one-hot` encodes string columns, and `--normalize-covar` centers and scales numeric covariates.

Common choices include ancestry PCs, age, sex, and study or processing batches. Choose covariates for the study
design rather than treating this list as required. If you use expression PCs in a cell-type-specific analysis,
compute them separately for each cell type.

jaxQTL adds an intercept by default. Use `--no-intercept` only when the design matrix already contains the intended
mean structure.

!!! warning "Duplicate sample IDs are rejected"

    Every input must identify a sample at most once. Duplicate IIDs make alignment ambiguous and cause setup to fail
    before mapping begins.

Sample inclusion files supplied with `--keep` or `--exclude` contain one IID per line and are applied to genotype
samples before the cross-input intersection.
