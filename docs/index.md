[![Documentation-webpage](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/jaxqtl)
[![Github](https://img.shields.io/github/stars/mancusolab/jaxqtl?style=social)](https://github.com/mancusolab/jaxqtl)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project generated with Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](
https://github.com/pypa/hatch
)

# jaxQTL
`jaxqtl` is a JAX-based toolkit for molecular QTL (molQTL) mapping. It implements fast generalized linear models (GLMs),
variant-level tests (Score/Wald with optional SPA correction), and gene-level multiple-testing procedures for cis scans.

  [**Installation**](#installation)
  | [**Quickstart**](#quickstart)
  | [**Notes**](#notes)
  | [**Support**](#support)

------------------

## Installation

For development installs:

``` bash
git clone https://github.com/mancusolab/jaxqtl.git
cd jaxqtl
python -m pip install -e .
```

## Quickstart

The CLI entrypoint is `jaxqtl`.

```bash
jaxqtl -h
```

### Cis mapping

Run cis mapping using a PLINK1 BED/BIM/FAM prefix (`--bfile`), plus phenotype and covariate tables.
`--geno` is deprecated; VCF (`--vcf`) is currently unsupported/experimental and is not a production genotype input.

```bash
jaxqtl cis \
  --bfile data/genotypes \
  --pheno data/expression.bed.gz \
  --covar data/covariates.tsv \
  --model nb \
  --test score \
  --window 500000 \
  --out results/jaxqtl
```

### Trans mapping

Stream trans results in genotype chunks:

```bash
jaxqtl trans \
  --bfile data/genotypes \
  --pheno data/expression.bed.gz \
  --covar data/covariates.tsv \
  --chunk-size 5000 \
  --out results/jaxqtl
```

See the API pages in the left navigation for details about GLMs, families/links, hypothesis tests, and mapping.

## Notes

- `jaxqtl` uses JAX JIT compilation for speed. For large scans, prefer running on CPU/GPU machines with adequate RAM.
- Offsets can be provided via an explicit file (`--offset`), a covariate (`--offset-name-from-covar`), or computed from
  library sizes (`--set-offset-from-libsize`) when available.

## Support

Please report bugs or feature requests in the GitHub issue tracker:
<https://github.com/mancusolab/jaxqtl/issues>

`jaxqtl` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.
