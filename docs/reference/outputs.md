# Output schemas

Mapping outputs are compressed Parquet files. Columns specific to the Negative Binomial model or permutation
calibration are omitted when they do not apply.

## Cis output

`jaxqtl cis` writes `${out}.cis.${test}.${perm|acat}.parquet.gz`.

| Group | Columns |
| --- | --- |
| Phenotype | `phenotype_id`, `chrom`, `num_var` |
| Lead variant | `snp`, `a1`, `a0`, `pos`, `tss_distance`, `af`, `ma_count` |
| Association | `beta`, `se`, `pvalue`, `pvalue_adj`, `adj_method` |
| Model | `nb_alpha`, `negloglikelihood`, `model_converged` |
| Validity | `result_valid`, `failure_reason` |
| Beta calibration | `shape1`, `shape2`, `nc_estimate`, `perm_converged` |

The Beta-calibration fields are not emitted for ACAT. `nb_alpha` is not emitted for Gaussian or Poisson models.
`negloglikelihood` is the fitted negative log-likelihood objective (lower is better). Score and SPA tests report the
shared null-model objective; Wald tests report the selected variant's full-model objective.

!!! note "Invalid rows preserve the tested phenotype"

    A gene with no finite SNP-level p-values remains in the cis output with `result_valid = false`. Its lead variant,
    association statistics, and convergence values are null. Use `failure_reason` to distinguish this state from a
    valid association with a large p-value.

## Nominal output

`jaxqtl nominal` writes `${out}.nominal.${test}.parquet.gz` with one row per phenotype–variant pair:

- `phenotype_id`, `chrom`, `snp`, `pos`, `a1`, `a0`, `tss_distance`, `af`, `ma_count`.
- `beta`, `se`, `pvalue`, `negloglikelihood`, `model_converged`.
- `nb_alpha` for Negative Binomial models.

For score and SPA tests, `negloglikelihood` is shared across all variants for a phenotype because they use one fitted
null model. For Wald tests, it is the fitted full-model objective for that phenotype–variant pair.

## Trans output

`jaxqtl trans` writes two files:

- `${out}.trans.${test}.variant.info.parquet.gz` contains `chrom`, `snp`, `pos`, `a1`, `a0`, `af`, and `ma_count`.
- `${out}.trans.${test}.sumstats.parquet.gz` contains `phenotype`, `snp`, `beta`, `se`, `pvalue`,
  `model_converged`, and `nb_alpha` for Negative Binomial models.

Within each phenotype block, summary-statistics rows follow the variant order in the metadata file.
