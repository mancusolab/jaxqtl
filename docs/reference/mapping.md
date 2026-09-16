# Mapping commands

`cis`, `nominal`, and `trans` share the options below. For worked commands, see
[Cis mapping](../guide/cis.md), [Nominal mapping](../guide/nominal.md), and [Trans mapping](../guide/trans.md).

## Invocation

```text
jaxqtl {cis,nominal,trans} GENOTYPE_SOURCE --pheno PATH --covar PATH [OPTIONS]
```

| Command | Tests | Gene-level calibration |
| --- | --- | --- |
| `cis` | Variants within each phenotype's cis window; reports a lead hit. | Beta permutation by default, or ACAT. |
| `nominal` | All associations within each phenotype's cis window. | None. |
| `trans` | All retained phenotype–variant pairs. | None. |

## Genotypes and phenotypes

Choose exactly one genotype source. See [Genotypes](../guide/genotypes.md) and
[Phenotypes](../guide/phenotypes.md) for file formats.

| Option | Default | Meaning |
| --- | --- | --- |
| `--bfile PREFIX` | None | PLINK1 BED/BIM/FAM prefix. |
| `--pfile PREFIX` | None | PLINK2 PGEN/PVAR/PSAM prefix. |
| `--vcf PATH` | None | Indexed VCF/BCF file. |
| `--bgen PATH` | None | BGEN file. |
| `--dosage` | Off | Read dosages instead of hard calls. |
| `--pheno PATH` | Required | BED-like or Parquet phenotype matrix. |
| `--covar PATH` | Required | Covariate table with sample IDs. |

The legacy `--geno` option is rejected; use an explicit genotype format.
Inputs are aligned by sample ID. Mapping retains phenotype chromosome labels shared with the genotype input.

## Covariates and offsets

| Option | Default | Meaning |
| --- | --- | --- |
| `--covar-name NAME...` | All columns | Include selected covariates. |
| `--rm-covar NAME...` | None | Exclude selected covariates; mutually exclusive with `--covar-name`. |
| `--one-hot` | Off | Encode string covariates as indicator columns, dropping one category. |
| `--normalize-covar` | Off | Center and scale numeric covariates. |
| `--no-intercept` | Off | Disable the automatically added intercept. |
| `--offset PATH` | None | Read fixed model offsets from a two-column sample-ID/offset TSV. |
| `--offset-name-from-covar NAME` | None | Extract a covariate column as the offset and remove it from fitted covariates. |
| `--set-offset-from-libsize` | Off | Compute log library sizes from loaded expression counts as offsets. |

The three offset options are mutually exclusive; no offset is used unless one is supplied.
External offsets are used as given, typically on the log scale. These differ from the raw sizes accepted by
`compute-pcs --libsize`. Computed library sizes preserve totals before gene selection.
See [Covariates](../guide/covariates.md) and [Offsets](../guide/offsets.md) for preparation examples.

## Sample, gene, and variant selection

| Option | Effect |
| --- | --- |
| `--keep PATH` / `--exclude PATH` | Retain or exclude sample IDs listed one per line; mutually exclusive. |
| `--genes ID...` / `--gene-list PATH` | Retain named genes, or genes listed one per line in a file. |
| `--rm-genes ID...` / `--exclude-gene-list PATH` | Exclude named genes, or genes listed one per line in a file. |
| `--chr LABEL` | Restrict to the exact chromosome label in expression metadata. |
| `--min-indiv-expr-pct FRACTION` | Retain samples expressing a fraction of selected genes strictly greater than this value. |
| `--min-gene-expr-pct FRACTION` | Retain genes expressed in a fraction of expression samples strictly greater than this value. |

Gene include/exclude options are mutually exclusive. Inline gene names may be comma- or space-delimited;
gene-list files have no header. Requested sample and gene IDs must exist in the expression file. Fractions lie
in `[0, 1]`; expression means a value greater than zero. The default gene threshold is `0`, removing genes with
no positive values. Sample expression QC is disabled unless requested.

| Mapping option | Default | Meaning |
| --- | --- | --- |
| `--maf FREQUENCY` | None | Exclude variants below this minor allele frequency. |
| `--window BP` | `500000` | Extend the cis interval by this many bases on each side. |
| `--tss-centered` | Off | Use TSS ± window instead of TSS − window through TES + window. |

For mapping, `--chr` must match an exact label present in both genotype and phenotype inputs.
`--window` and `--tss-centered` apply only to `cis` and `nominal`.
Mapping selects genes/chromosomes and applies gene expression QC before sample expression QC and final input
alignment. PCA has its own [processing order](compute-pcs.md#processing-order).

## Models and tests

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `nb` | Response model: `nb` (Negative Binomial), `poisson`, or `gaussian`. |
| `--test` | `score` | Variant test: `score` or `wald`. |
| `--robust-se` | Off | Huber–White sandwich standard errors; requires `--test wald`. |
| `--spa` | Off | Saddlepoint correction for score-test p-values in count models. |
| `--acat` | Off | Aggregate variant p-values with ACAT in `cis`. |
| `--nperm INT` | `1000` | Permutations for Beta calibration in `cis` when ACAT is disabled. |

The default test is `score` for all three commands. Specify `--test wald` when a nominal scan
needs full-model effect estimates. SPA is skipped for Gaussian models and does not apply to Wald tests.
`--acat` and `--nperm` do not calibrate nominal or trans results.

For score-test ACAT scans, **`--spa --acat` is strongly recommended** because ACAT is sensitive to
variant p-value calibration. See [Tests and gene-level calibration](../guide/tests.md#tail-and-gene-level-calibration)
for the statistical choices and [Models](../guide/model.md) for response assumptions.

## GLM fitting controls

| Option | Default | Meaning |
| --- | --- | --- |
| `--max-iter` | `1000` | Maximum IRLS iterations |
| `--tol` | `1e-3` | Absolute change in total negative log likelihood that triggers the gradient check |
| `--gtol` | `1e-3` | Per-observation gradient tolerance, with coefficient scaling and NB2 bound projection |
| `--step-size` | `1.0` | Initial trial step for each IRLS update; rejected trials are halved |
| `--solver` | `cholesky` | Weighted least-squares solver; choices are `cholesky`, `qr`, and `cg` |

Both likelihood and gradient criteria must be met for GLM convergence. These controls govern model fitting;
they do not set the SPA root-solver or Beta-calibration tolerances. See
[Troubleshooting](../guide/troubleshooting.md) for interpretation and troubleshooting.

## Runtime and outputs

| Option | Default | Meaning |
| --- | --- | --- |
| `--seed INT` | `0` | Random seed. |
| `--platform`, `-p` | `cpu` | JAX backend: `cpu`, `gpu`, or `tpu`. |
| `--verbose` | Off | Enable debug logging. |
| `--out PREFIX`, `-o PREFIX` | `jaxqtl` | Prefix for results and the log file. |
| `--help`, `-h` | — | Print command help and exit. |

See [Output schemas](outputs.md) for filenames, columns, and failed-test handling.
