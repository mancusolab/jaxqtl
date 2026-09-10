# jaxQTL

jaxQTL is a JAX-based command-line tool and Python library for cell-type-specific eQTL mapping from donor-level
single-cell pseudobulk expression. It provides Poisson and Negative Binomial count models, while also supporting
Gaussian molecular phenotypes.

[Try the quickstart](guide/quickstart.md){ .md-button .md-button--primary }
[Install jaxQTL](guide/installation.md){ .md-button }
[Browse the Python API](api/models/glm.md){ .md-button }

## Install

```bash
pip install jaxqtl
jaxqtl --help
```

## Run the tutorial

The [Quickstart](guide/quickstart.md) runs ten genes from the bundled pseudobulk data and shows how to inspect the
results. Choose **SPA + ACAT** for fast gene-level testing without permutations, or **Beta permutation** for
permutation calibration. SPA is strongly recommended with score-test ACAT because ACAT is sensitive to variant
p-value calibration.

To analyze your own data, follow the [single-cell workflow](guide/single-cell-cis.md) from pseudobulk preparation
through multiple-testing correction. The task guides cover [cis](guide/cis.md), [nominal](guide/nominal.md),
[trans](guide/trans.md), and [expression-PC](guide/compute-pcs.md) workflows.

## Support and citation

Report bugs and feature requests through the [GitHub issue tracker](https://github.com/mancusolab/jaxqtl/issues).
See [Citation](cite.md) when using jaxQTL in published work.
