[![PyPI](https://img.shields.io/pypi/v/jaxqtl.svg)](https://pypi.org/project/jaxqtl/)
[![GitHub stars](https://img.shields.io/github/stars/mancusolab/jaxqtl?style=social)](https://github.com/mancusolab/jaxqtl)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# jaxQTL

jaxQTL is a JAX-based command-line tool and Python library for cell-type-specific eQTL mapping from donor-level
single-cell pseudobulk expression. It provides Poisson and Negative Binomial count models and also supports Gaussian
molecular phenotypes.

**For fast gene-level cis testing, we recommend SPA + ACAT (`--spa --acat`).** This combines saddlepoint-adjusted
score-test p-values with the aggregated Cauchy association test, avoiding permutation refits. It is typically much
faster than permutation calibration; the speedup depends on the data and permutation count.

>   Zhang, Z., Kim, A., Suboc, N., Mancuso, N., and Gazal, S. (2025). Efficient count-based models improve power and robustness for large-scale single-cell eQTL mapping. medRxiv (https://www.medrxiv.org/content/10.1101/2025.01.18.25320755v2)

[Read the documentation](https://mancusolab.github.io/jaxqtl/)

## Installation

jaxQTL requires Python 3.11 or newer.

```bash
pip install jaxqtl
jaxqtl --help
```

## Quick example

From a repository checkout, run a SPA + ACAT cis scan over ten genes from the bundled tutorial data:

```bash
jaxqtl cis \
  --bfile tutorial/input/chr22_N100 \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --covar tutorial/input/donor_features.tsv \
  --gene-list tutorial/input/genelist_10 \
  --model nb \
  --test score \
  --set-offset-from-libsize \
  --normalize-covar \
  --spa \
  --acat \
  --out tutorial/output/quickstart_spa_acat
```

The command writes `tutorial/output/quickstart_spa_acat.cis.score.spa.acat.parquet.gz`.

> [!WARNING]
> **Use SPA with score-test ACAT.** ACAT can amplify inaccurate variant tail p-values into misleading gene-level
> results. We strongly recommend using both flags; SPA can still fall back to the normal approximation.

For your own data, compute library-size offsets from the full gene matrix; see
[Offsets](https://mancusolab.github.io/jaxqtl/guide/offsets/) before using a restricted input file.
Check convergence and apply FDR correction across genes with either method.

Beta permutation remains available when a permutation reference is desired: replace `--spa --acat` with
`--nperm 1000`. The methods can yield different discoveries; see
[Calibration tradeoffs](https://mancusolab.github.io/jaxqtl/guide/tests/#tail-and-gene-level-calibration).

The documentation covers the [single-cell cis-eQTL workflow](https://mancusolab.github.io/jaxqtl/guide/single-cell-cis/),
the [tutorial](https://mancusolab.github.io/jaxqtl/guide/quickstart/),
[input formats](https://mancusolab.github.io/jaxqtl/reference/inputs/),
[mapping workflows](https://mancusolab.github.io/jaxqtl/guide/cis/), and
[output interpretation](https://mancusolab.github.io/jaxqtl/guide/postprocessing/).

## Development

See the [development setup](https://mancusolab.github.io/jaxqtl/contributing/) for environment, test, and
documentation build instructions.

## Citation and support

See the [citation page](https://mancusolab.github.io/jaxqtl/cite/) when using jaxQTL in published work. Report bugs
and feature requests through the [GitHub issue tracker](https://github.com/mancusolab/jaxqtl/issues).
