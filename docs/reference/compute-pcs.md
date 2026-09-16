# Expression PCA

`jaxqtl compute-pcs` estimates expression principal components and saves their explained-variance proportions.
For worked commands, see [Compute expression PCs](../guide/compute-pcs.md).

## Invocation and inputs

```text
jaxqtl compute-pcs --pheno PATH --num-pcs N [OPTIONS]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--pheno PATH` | Required | BED-like or Parquet expression matrix; see [Phenotypes](../guide/phenotypes.md). |
| `--num-pcs N` | Required | Positive component count, subject to the limits below. |
| `--covar PATH` | None | Intersect samples before fitting and append PCs to this covariate table. |

No genotype input is needed. With `--covar`, only shared samples remain, in expression-file order.
Covariates are copied without encoding, scaling, or residualizing expression. Columns beginning with
`ExprPC` are rejected because they conflict with the output names. Sample identifiers must be unique and nonmissing.

## Sample and gene selection

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

## Normalization and transformation

| Option | Default | Choices or input |
| --- | --- | --- |
| `--normalization` | `library-size` | `library-size`, `tmm`, `none` |
| `--transform` | `log1p` | `log1p`, `none` |
| `--libsize PATH` | Loaded count totals | TSV with raw library sizes. |
| `--libsize-name-from-covar NAME` | None | Raw library-size column in `--covar`; mutually exclusive with `--libsize`. |

The defaults are `--normalization library-size --transform log1p`, intended for raw counts. Normalization and
transformation are separate stages:

| Normalization | Relative size factor applied before transformation |
| --- | --- |
| `library-size` (default) | `l / median(l)` using each sample's library size `l` |
| `tmm` | `e / median(e)` where `e = l * TMM_factor` is the effective library size |
| `none` | No scaling; library sizes are not required |

`--transform log1p` (default) applies `log(1 + y)` to the normalized values. `--transform none` skips this step.
All modes subsequently center and scale variable genes for PCA. Counts must be nonnegative when normalization or
`log1p` is enabled.

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

## Processing order

1. Select samples and intersect optional covariate IDs, preserving expression-file order.
2. Use library-size totals across all genes in the loaded file, or an external source.
3. Apply sample expression QC using the full input gene set.
4. Normalize expression, estimating TMM factors from nonzero genes when requested.
5. Select genes/chromosomes and apply gene expression QC using the retained samples.
6. Apply the log transform if requested, remove constant genes, and standardize remaining genes.
7. Validate the requested component count and fit PCA.

## Component counts and reproducibility

`--num-pcs` is required and must be positive. After filtering and transformation, at least two samples and one
variable gene must remain, and the component count cannot exceed `min(n_samples - 1, n_variable_genes)`.
Nonnumeric/nonfinite expression and duplicate or missing sample IDs are rejected.

Components are ordered by decreasing explained variance within the estimated subspace. The algorithm estimates
that subspace iteratively, then uses a small projected SVD to resolve its principal directions. Reusing `--seed`
with the same inputs makes initialization reproducible, although floating-point results can vary across JAX backends.

## Runtime and output options

| Option | Default | Meaning |
| --- | --- | --- |
| `--seed INT` | `0` | Random initialization seed. |
| `--platform`, `-p` | `cpu` | JAX backend: `cpu`, `gpu`, or `tpu`. |
| `--verbose` | Off | Enable debug logging. |
| `--out PATH`, `-o PATH` | `jaxqtl.princ_comp.tsv` | Output TSV filename, including its extension. |
| `--help`, `-h` | — | Print command help and exit. |

## Outputs

For `--out pcs.tsv`, the command writes:

- `pcs.tsv`: sample IDs and PCs, with original covariates first when supplied.
- `pcs.tsv.variance.tsv`: `component`, `explained_variance_ratio`, and `cumulative_explained_variance_ratio`.
- `pcs.tsv.log`: progress, sample/gene filtering summaries, and each component's explained-variance proportion.

Use the variance table directly for a scree plot. Ratios describe the transformed, gene-standardized matrix,
with total variance across all components as the denominator. They need not sum to one when only some components
are requested. The PC columns themselves have unit norm; their column variances do not measure variance explained.


For Python use, see [ExpressionData](../api/data/pheno.md), including `normalize` and `compute_pcs`.
