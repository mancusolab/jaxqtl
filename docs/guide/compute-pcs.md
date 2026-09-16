# Compute expression PCs

`jaxqtl compute-pcs` estimates expression principal components from a BED-like or Parquet expression matrix.
Expression PCs are optional covariates; choose whether to include them as part of the study design.

## Compute PCs from raw counts

```bash
jaxqtl compute-pcs \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --num-pcs 2 \
  --normalization library-size \
  --transform log1p \
  --out tutorial/output/CD4_NC.N100.expr_pcs.tsv
```

No genotype or covariate file is required. The output has `iid`, `ExprPC1`, `ExprPC2`, and subsequent requested
components, with samples in expression-file order. Add `--covar PATH` to append PCs to an existing covariate table.
The command intersects expression and covariate sample IDs **before fitting**, logs unmatched samples, and writes
only retained samples. Covariate values are copied without encoding, scaling, or residualizing expression.
Existing covariate columns beginning with `ExprPC` are rejected to prevent ambiguous output.

For cell-type-specific analyses, run the command separately for each cell type's pseudobulk matrix.

## Select samples and genes

These selection options are also available in the mapping commands:

| Option | Effect |
| --- | --- |
| `--keep PATH` / `--exclude PATH` | Retain or exclude sample IDs listed one per line; mutually exclusive. |
| `--genes ID...` / `--gene-list PATH` | Retain named genes, or genes listed one per line in a file. |
| `--rm-genes ID...` / `--exclude-gene-list PATH` | Exclude named genes, or genes listed one per line in a file. |
| `--chr LABEL` | Restrict to the exact chromosome label in expression metadata. |
| `--min-indiv-expr-pct FRACTION` | Retain samples expressing a fraction of input genes strictly greater than this value. |
| `--min-gene-expr-pct FRACTION` | Retain genes expressed in a fraction of retained samples strictly greater than this value. |

Gene include/exclude options are mutually exclusive. Inline gene names may be comma- or space-delimited;
gene-list files have no header. Requested sample and gene IDs must exist in the expression file. Fractions lie
in `[0, 1]`; expression means a value greater than zero. The default gene threshold is `0`, removing genes with
no positive values. Sample expression QC is disabled unless requested.

For PCA, preprocessing follows this order:

1. Select samples and intersect optional covariate IDs, preserving expression-file order.
2. Use library-size totals across all genes in the loaded file, or an external source.
3. Apply sample expression QC using the full input gene set.
4. Normalize expression, estimating TMM factors from nonzero genes when requested.
5. Select genes/chromosomes and apply gene expression QC using the retained samples.
6. Apply the log transform if requested, remove constant genes, and standardize remaining genes.
7. Validate the requested component count and fit PCA.

For example:

```bash
jaxqtl compute-pcs \
  --pheno expression.bed.gz \
  --covar covariates.tsv \
  --keep analysis_samples.txt \
  --gene-list pca_genes.txt \
  --min-indiv-expr-pct 0.1 \
  --min-gene-expr-pct 0.1 \
  --num-pcs 10 \
  --normalization library-size \
  --transform log1p \
  --out covariates_with_pcs.tsv
```

## Normalization and library sizes

The defaults are `--normalization library-size --transform log1p`, intended for raw counts. Normalization and
transformation are separate stages:

| Normalization | Relative size factor applied before transformation |
| --- | --- |
| `library-size` (default) | `l / median(l)` using each sample's library size `l` |
| `tmm` | `e / median(e)` where `e = l * TMM_factor` is the effective library size |
| `none` | No scaling; library sizes are not required |

`--transform log1p` (default) applies `log(1 + y)` to the normalized values. `--transform none` skips this step.
All modes subsequently center and scale variable genes for PCA. Counts must be nonnegative when normalization or
`log1p` is enabled. The old combined `lognorm` option is not supported.

Examples:

```bash
# Library-size normalization followed by log1p (the defaults)
--normalization library-size --transform log1p

# TMM composition adjustment followed by log1p
--normalization tmm --transform log1p

# Already normalized expression that still needs a log transform
--normalization none --transform log1p

# Expression already normalized and log-transformed
--normalization none --transform none
```

Automatic library sizes include all genes in the loaded file, even genes later removed by selection or QC.
TMM factors are estimated after sample selection and sample QC, using all remaining nonzero input genes before
PCA gene/chromosome selection. The median reference size is calculated across the samples retained for PCA.
TMM followed by `log1p` uses the same median-based scale as library-size normalization; it is not edgeR log-CPM.
TMM trimming uses ranked log-ratios. Values tied within floating-point roundoff can fall on different sides of
a trimming boundary in JAX and edgeR, so exact cross-implementation agreement is not guaranteed for such inputs.

If the input file was already restricted to a subset of genes or chromosomes, supply library sizes computed
from the original count matrix. Choose one of:

- `--libsize PATH`: a TSV with exactly `iid` and `libsize` columns (the usual `IID`/`#IID` aliases are accepted).
- `--libsize-name-from-covar NAME`: a column from the table supplied with `--covar`. This column remains in output.

Both options require `library-size` or `tmm` normalization. These are **raw positive library sizes**, not log-scale
model offsets. Sizes are aligned by sample ID. Every retained sample needs one finite, positive size; extra samples
are ignored. Duplicate or missing sample IDs are rejected. External library sizes cannot restore missing genes for
TMM factor estimation: provide the full expression input and use the CLI gene filters when possible.

## Outputs and scree plots

For `--out pcs.tsv`, the command writes:

- `pcs.tsv`: sample IDs and PCs, with original covariates first when supplied.
- `pcs.tsv.variance.tsv`: `component`, `explained_variance_ratio`, and `cumulative_explained_variance_ratio`.
- `pcs.tsv.log`: progress, sample/gene filtering summaries, and each component's explained-variance proportion.

Use the variance table directly for a scree plot. Ratios describe the transformed, gene-standardized matrix,
with total variance across all components as the denominator. They need not sum to one when only some components
are requested. The PC columns themselves have unit norm; their column variances do not measure variance explained.

The Python method returns both the component table and a NumPy array of proportions in matching component order:

```python
pcs, explained_variance_ratio = expression_data.compute_pcs(10, rng_key, normalization="library-size", transform="log1p")
```

For TMM with a restricted PCA gene set in Python, normalize before selecting genes, then disable normalization
in `compute_pcs` to avoid applying it twice:

```python
normalized = expression_data.normalize("tmm")
selected = normalized.filter_genes_by_ids(keep=pca_genes)
pcs, explained_variance_ratio = selected.compute_pcs(10, rng_key, normalization="none", transform="log1p")
```

## Component counts and reproducibility

`--num-pcs` is required and must be positive. After filtering and transformation, at least two samples and one
variable gene must remain, and the component count cannot exceed `min(n_samples - 1, n_variable_genes)`.
Nonnumeric/nonfinite expression and duplicate or missing sample IDs are rejected.

Components are ordered by decreasing explained variance within the estimated subspace. The algorithm estimates
that subspace iteratively, then uses a small projected SVD to resolve its principal directions. Reusing `--seed`
with the same inputs makes initialization reproducible, although floating-point results can vary across JAX backends.
