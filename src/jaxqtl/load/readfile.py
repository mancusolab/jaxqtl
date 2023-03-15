from typing import NamedTuple, Optional

import pandas as pd
import scanpy as sc
from anndata._core.anndata import AnnData
from pandas_plink import read_plink1_bin

import jax.numpy as jnp


class PlinkState(NamedTuple):
    bed: jnp.ndarray
    bim: jnp.ndarray
    fam: jnp.ndarray


class RawDataState(NamedTuple):
    genotype: jnp.ndarray
    count: AnnData


def readraw(
    geno_path: str,
    pheno_path: str,
    cov_path: Optional[str],
    min_genes: int = 200,
    min_cells: int = 3,
):
    """
    Genotype data: plink file
    pheno_path: h5ad file path, including covariates

    Gene expression data: h5ad file
    - dat.X: cell x gene sparse matrix, where cell is indexed by unique barcode
    - dat.obs: cell x features (eg. donor_id, age,...)
    - dat.var: gene x gene summary stats
    -

    recode sex as: female = 1, male = 0
    """
    # Append prefix with suffix
    bed_path = geno_path + ".bed"
    bim_path = geno_path + ".bim"
    fam_path = geno_path + ".fam"

    G = read_plink1_bin(bed_path, bim_path, fam_path, verbose=False)
    genotype = jnp.array(G.values)  # sample x variants

    # read age, sex and PCs from plink PCA result
    cov = pd.read_csv(cov_path, sep=" ", header=None)
    cov = jnp.array(cov)[:, 2:]

    dat = sc.read_h5ad(pheno_path)

    # pre-process steps: no normalization
    # https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
    # for each gene in a given cell, take mean of

    # filter cells by min number of genes expressed (in place)
    sc.pp.filter_cells(dat, min_genes=min_genes)
    # filter genes by min number of cells expressed (in place)
    sc.pp.filter_genes(dat, min_cells=min_cells)
    # filter by mitochondria

    # a0=0, a1=1, genotype value (0/1/2) is the count for a1 allele
    # print(G.a0.sel(variant="variant0").values)
    # print(G.sel(sample="1", variant="variant0").values)

    return RawDataState(genotype, dat)
