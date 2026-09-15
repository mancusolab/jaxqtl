# Command-line interface

The `jaxqtl` executable provides four subcommands:

| Command | Result |
| --- | --- |
| `jaxqtl compute-pcs` | Expression principal components, optional appended covariates, and explained-variance table |
| `jaxqtl cis` | One lead association and adjusted p-value per tested phenotype |
| `jaxqtl nominal` | Every association within each cis window |
| `jaxqtl trans` | Chunked phenotype-by-variant associations |

Run `jaxqtl COMMAND --help` for the complete parser-generated option list and defaults.

## Common mapping options

All mapping commands require one genotype source plus `--pheno` and `--covar`.

| Group | Options |
| --- | --- |
| Genotypes | `--bfile`, `--pfile`, `--vcf`, `--bgen`, `--dosage` |
| Covariates | `--covar-name`, `--rm-covar`, `--normalize-covar`, `--one-hot`, `--no-intercept` |
| Library-size adjustment (offsets) | `--offset`, `--offset-name-from-covar`, `--set-offset-from-libsize` |
| Model and variant testing | `--model`, `--test`, `--robust-se`, `--spa` |
| Gene-level testing | `--acat`, `--nperm` |
| Filters | `--keep`, `--exclude`, `--min-indiv-expr-pct`, `--min-gene-expr-pct`, `--maf`, `--chr` |
| Phenotypes | `--gene-list`, `--genes`, `--rm-genes`, `--exclude-gene-list`, `--window`, `--tss-centered` |
| Solver | `--max-iter`, `--tol`, `--gtol`, `--step-size`, `--solver` |
| Runtime | `--seed`, `--platform`, `--verbose`, `--out` |

Some accepted flags apply only to particular combinations. `--robust-se` requires a Wald test; `--spa` applies to
score tests; `--acat` and `--nperm` affect only `cis`; and `--window` and `--tss-centered` affect only `cis` and
`nominal`.

For score-test ACAT scans, **`--spa --acat` is strongly recommended** because ACAT is sensitive to variant
p-value calibration. See [Tests and gene-level calibration](../guide/tests.md#tail-and-gene-level-calibration)
for why Beta permutation does not have the same dependence on asymptotic tail probabilities.

Mapping automatically retains expression phenotypes on chromosome labels shared with the genotype input. `--chr`
further restricts both phenotypes and genotype variants to one exact label, which must occur in both inputs.

See the [workflow guides](../guide/quickstart.md) for complete commands with compatible options.

## Expression PCA

`compute-pcs` requires `--pheno` and `--num-pcs`. It supports the sample and gene filters listed above, including
`--chr`, without requiring genotype inputs. `--covar` is optional; when supplied, sample IDs are intersected before
PCA and the retained covariates are included in output.

Use `--transform lognorm` for raw counts. Library sizes default to totals across all loaded genes before filtering;
override them with `--libsize PATH` or `--libsize-name-from-covar NAME`. These options take raw positive library sizes,
not mapping offsets. `log1p` and no transform are also supported.

See [Compute expression PCs](../guide/compute-pcs.md) for filtering order, normalization, and scree-plot outputs.

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
