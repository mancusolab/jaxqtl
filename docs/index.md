# jaxQTL

jaxQTL is a JAX-based command-line tool and Python library for cell-type-specific eQTL mapping from donor-level
single-cell pseudobulk expression. It provides Poisson and Negative Binomial count models, while also supporting
Gaussian molecular phenotypes.

[Run single-cell cis-eQTL](guide/single-cell-cis.md){ .md-button .md-button--primary }
[Install jaxQTL](guide/installation.md){ .md-button }
[Browse the Python API](api/models/glm.md){ .md-button }

## Install

```bash
pip install jaxqtl
jaxqtl --help
```

## Run the tutorial

The bundled example is a donor-by-gene pseudobulk matrix for one cell type. From a repository checkout:

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

The [single-cell workflow](guide/single-cell-cis.md) explains how to prepare cell-type-specific inputs. The
[quickstart](guide/quickstart.md) explains the bundled data, offset choice, and output. Use the task guides for
[cis](guide/cis.md), [nominal](guide/nominal.md), [trans](guide/trans.md), and
[expression-PC](guide/compute-pcs.md) workflows.

## Support and citation

Report bugs and feature requests through the [GitHub issue tracker](https://github.com/mancusolab/jaxqtl/issues).
See [Citation](cite.md) when using jaxQTL in published work.
