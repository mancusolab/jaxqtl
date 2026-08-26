# Command-line interface

The `jaxqtl` executable provides four subcommands:

| Command | Result |
| --- | --- |
| `jaxqtl compute-pcs` | Expression principal components appended to covariates |
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
| Phenotypes | `--gene-list`, `--genes`, `--window`, `--tss-centered` |
| Solver | `--max-iter`, `--tol`, `--step-size`, `--solver` |
| Runtime | `--seed`, `--platform`, `--verbose`, `--out` |

Some accepted flags apply only to particular combinations. `--robust-se` requires a Wald test; `--spa` applies to
score tests; `--acat` and `--nperm` affect only `cis`; and `--window` and `--tss-centered` affect only `cis` and
`nominal`.

Mapping automatically retains expression phenotypes on chromosome labels shared with the genotype input. `--chr`
further restricts both phenotypes and genotype variants to one exact label, which must occur in both inputs.

See the [workflow guides](../guide/quickstart.md) for complete commands with compatible options.
