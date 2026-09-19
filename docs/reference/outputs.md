# Output schemas

Mapping outputs are compressed Parquet files. Columns specific to the Negative Binomial model or permutation
calibration are omitted when they do not apply.

## Expression PCA output

`jaxqtl compute-pcs --out PATH` writes a TSV at `PATH` with `iid`, optional covariates, and `ExprPC1` through
`ExprPC{num_pcs}` in decreasing explained-variance order. Rows follow retained expression-file sample order.

`PATH.variance.tsv` contains one row per component:

| Column | Meaning |
| --- | --- |
| `component` | Matching PC column name, such as `ExprPC1` |
| `explained_variance_ratio` | Proportion of total transformed, standardized expression variance explained |
| `cumulative_explained_variance_ratio` | Sum of proportions through this component |

The command also logs these proportions to `PATH.log`. See [Expression PCA reference](compute-pcs.md)
for preprocessing and cohort-selection details.

## Shared mapping columns

Cis, nominal, and trans results use the same names for association statistics and diagnostics:

| Column | Meaning |
| --- | --- |
| `phenotype_id` | Tested phenotype identifier. |
| `snp` | Variant identifier; the selected lead variant in cis output. |
| `beta`, `se`, `pvalue` | Variant effect, standard error, and nominal p-value. |
| `negloglikelihood` | Fitted negative log-likelihood objective (lower is better). |
| `model_converged` | Whether the fitted model met its convergence criteria. |
| `result_valid` | Whether the reported numerical association fields pass the checks below. |
| `failure_reason` | First failed numerical check, or null for a valid result. |
| `nb_alpha` | Negative Binomial dispersion; omitted for Gaussian and Poisson models. |

Score and SPA tests report the shared null-model objective and convergence flag. Wald tests report each
variant's full-model objective and convergence flag. For cis output, these fields describe the selected lead.

### Validity and convergence

`result_valid` requires a finite effect, a finite positive standard error, a finite p-value in `[0, 1]`, and a
finite objective. When present, dispersion must be finite and nonnegative, and the adjusted p-value must be
finite and in `[0, 1]`. Zero p-values are accepted. Invalid association rows are retained with their computed
values, including NaNs or infinities, so the failure can be inspected.

Validity is separate from convergence: a finite but nonconverged fit can have `result_valid = true` and
`model_converged = false`. Downstream analyses should check both flags, plus `perm_converged` when present.
These checks do not establish model adequacy or statistical calibration.

When several checks fail, `failure_reason` reports the first applicable reason in this order:

| Reason | Condition |
| --- | --- |
| `invalid_standard_error` | Standard error is missing, nonfinite, or nonpositive. |
| `nonfinite_effect` | Effect is missing or nonfinite. |
| `invalid_pvalue` | Nominal p-value is missing, nonfinite, or outside `[0, 1]`. |
| `nonfinite_objective` | Fitted objective is missing or nonfinite. |
| `invalid_dispersion` | NB dispersion is missing, nonfinite, or negative. |
| `invalid_adjusted_pvalue` | Cis adjusted p-value is missing, nonfinite, or outside `[0, 1]`. |

Cis additionally uses `no_finite_pvalues` when no lead variant can be selected. Such a row retains the phenotype
identifier and variant count, but its lead, association, and convergence fields are null.

## Cis output

`jaxqtl cis` writes `${out}.cis.${test}.${perm|acat}.parquet.gz`.

Each row contains the shared mapping columns, variant metadata (`chrom`, `pos`, `a1`, `a0`, `tss_distance`, `af`,
`ma_count`), and these gene-level fields:

| Column | Meaning |
| --- | --- |
| `num_var` | Number of variants tested in the cis window. |
| `pvalue_adj` | Gene-level adjusted p-value; this is not study-level FDR. |
| `adj_method` | `ACAT` or `BETA`. |

Beta-permutation calibration adds `shape1`, `shape2`, `nc_estimate`, and `perm_converged`. These fields are omitted
for ACAT. `perm_converged` reports convergence of both the Beta fit and the reference estimate.

## Nominal output

`jaxqtl nominal` writes `${out}.nominal.${test}.parquet.gz` with one row per phenotype–variant pair.

It contains the shared mapping columns and the same variant metadata as cis: `chrom`, `pos`, `a1`, `a0`,
`tss_distance`, `af`, and `ma_count`. It omits the gene-level and Beta-calibration fields.

## Trans output

`jaxqtl trans` writes two files:

- `${out}.trans.${test}.variant.info.parquet.gz` contains `chrom`, `snp`, `pos`, `a1`, `a0`, `af`, and `ma_count`.
- `${out}.trans.${test}.sumstats.parquet.gz` contains the shared mapping columns, including `phenotype_id`,
  `negloglikelihood`, `result_valid`, and `failure_reason`.

Variant metadata remain separate to avoid repeating them for every phenotype. Within each phenotype block,
summary-statistics rows follow the variant order in the metadata file. Trans output has no cis distance or
gene-level calibration columns.

### Changes from 0.3.1

Trans summary statistics now use `phenotype_id` in place of `phenotype`. Nominal and trans output gain
`result_valid` and `failure_reason`; trans also gains `negloglikelihood`. Cis uses the same numerical checks
instead of treating every selected lead as valid. Read Parquet columns by name: cis column ordering now follows
nominal output before adding gene-level and permutation fields.
