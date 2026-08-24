# jaxQTL

jaxQTL is a JAX-based command-line tool and Python library for molecular QTL mapping with Gaussian, Poisson, and
Negative Binomial models. It supports score and Wald tests, saddlepoint tail calibration, gene-level permutation or
ACAT procedures, and chunked trans scans.

[Get started](guide/installation.md){ .md-button .md-button--primary }
[View the workflows](guide/quickstart.md){ .md-button }
[Browse the Python API](api/models/glm.md){ .md-button }

## Install

```bash
pip install jaxqtl
jaxqtl --help
```

## Run the tutorial

From a repository checkout:

```bash
jaxqtl cis \
  --bfile tutorial/input/chr22_N100 \
  --pheno tutorial/input/CD4_NC.N100.bed.gz \
  --covar tutorial/input/donor_features.tsv \
  --gene-list tutorial/input/genelist_10 \
  --model nb \
  --test score \
  --set-offset-from-libsize \
  --nperm 1000 \
  --out tutorial/output/quickstart
```

The [quickstart](guide/quickstart.md) explains the inputs, offset choice, and output. Use the task guides for
[cis](guide/cis.md), [nominal](guide/nominal.md), [trans](guide/trans.md), and
[expression-PC](guide/compute-pcs.md) workflows.

## Support and citation

Report bugs and feature requests through the [GitHub issue tracker](https://github.com/mancusolab/jaxqtl/issues).
See [Citation](cite.md) when using jaxQTL in published work.
