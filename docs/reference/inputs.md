# Input formats

## Genotypes

Provide one genotype source per command:

- `--bfile PREFIX`: `PREFIX.bed`, `PREFIX.bim`, and `PREFIX.fam`.
- `--pfile PREFIX`: `PREFIX.pgen`, `PREFIX.pvar`, and `PREFIX.psam`.
- `--vcf PATH`: indexed VCF or BCF.
- `--bgen PATH`: BGEN.

Variant metadata must provide chromosome, position, identifier, and `a0`/`a1` alleles through the selected `genoio`
adapter. jaxQTL computes allele frequency and minor-allele count from the loaded genotype values.

## Phenotypes

Phenotypes are read from `.bed`, `.bed.gz`, `.parquet`, or `.parquet.gz`. The first four columns are chromosome,
start, end, and phenotype ID. Remaining columns are sample IDs with phenotype values.

The [phenotype guide](../guide/phenotypes.md) lists accepted metadata aliases.

## Covariates

Covariates are tab-delimited, with exactly one IID-like column and at least one data column. `iid`, `IID`, `#iid`, and
case variants are accepted and normalized internally to `iid`. An FID-like column is ignored.

The reader recognizes `NA`, an empty field, `NULL`, `NaN`, and `nan` as missing values.

## Offsets

An offset file is a tab-delimited table containing one IID-like column and one numeric log-offset column. If the input
has multiple data columns, choose one by name with the Python reader or prepare a two-column file for the CLI.

See [Offsets](../guide/offsets.md) for scale and preprocessing requirements.

## Lists

Gene, keep-sample, and exclude-sample files contain one identifier per line. A first line beginning with `#` is
treated as a header and skipped.
