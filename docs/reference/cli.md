# Command-line overview

Run `jaxqtl COMMAND [OPTIONS]`. Choose a command by the result you need:

| Command | Required inputs | Result |
| --- | --- | --- |
| `cis` | Genotypes, phenotypes, covariates | Lead association and gene-level adjusted p-value per tested phenotype. |
| `nominal` | Genotypes, phenotypes, covariates | Every association within each cis window. |
| `trans` | Genotypes, phenotypes, covariates | Associations across retained phenotypes and variants, written in chunks. |
| `compute-pcs` | Expression matrix, component count | Expression PCs and their explained-variance proportions. |

## Command reference

- [Mapping commands](mapping.md): input selection, covariates, offsets, tests, filters, and fitting controls.
- [Expression PCA](compute-pcs.md): normalization, transformation, filtering order, component limits, and output files.
- [Output schemas](outputs.md): mapping result columns and PCA variance tables.

For complete workflows, start with the [Quickstart](../guide/quickstart.md) or
[Compute expression PCs](../guide/compute-pcs.md).

## Help and option conventions

```bash
jaxqtl --help
jaxqtl cis --help
jaxqtl nominal --help
jaxqtl trans --help
jaxqtl compute-pcs --help
```

Options follow the subcommand. Help lists accepted values and defaults.
Boolean flags such as `--verbose` take no value. Options accepting several names, such as `--genes`,
accept comma- or space-delimited names; identifier files contain one ID per line without a header.

All commands support `--seed` (default `0`), `--platform` (default `cpu`), and `--verbose`.
GPU and TPU execution require a compatible JAX installation; see [Installation](../guide/installation.md).

## Output paths

Mapping treats `--out` as a **prefix** (default `jaxqtl`) and adds command-specific result suffixes.
Expression PCA treats it as a **TSV filename** (default `jaxqtl.princ_comp.tsv`) and adds
`.variance.tsv` for its companion table. Every command writes a log at `<out>.log`.
