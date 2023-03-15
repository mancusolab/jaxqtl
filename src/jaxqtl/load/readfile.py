from typing import NamedTuple

import decoupler as dc
import numpy as np
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


class CleanDataState(NamedTuple):
    """
    count: filtered cells and genes for given cell type, contains sample features
    """

    genotype: jnp.ndarray
    count: AnnData


def process_count(dat: AnnData, cell_type: str = "CD14-positive monocyte") -> AnnData:
    """
    dat: n_obs (cell) x n_vars (genes)
    dat.var_name = 'ensembl_id'

    No normalization on the count data!

    ref: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
    """
    # start with: 11043 x 36571
    # filter cells by min number of genes expressed (in place)
    sc.pp.filter_cells(dat, min_genes=200)  # 11043 x 36571
    # filter genes by min number of cells expressed (in place)
    sc.pp.filter_genes(dat, min_cells=3)  # 11043 × 17841

    # show top 20 genes highly expressed in each cell across all cells
    # sc.pl.highest_expr_genes(dat, n_top=20, )
    # sc.pl.scatter(dat, x='nCount_RNA', y='percent.mt')
    # sc.pl.scatter(dat, x='nCount_RNA', y='nFeature_RNA')

    #  filter cells with too many genes expressed
    dat = dat[dat.obs["nFeature_RNA"] < 2500, :]  # 11026 × 17841

    #  filter cells that have >5% mitochondrial counts
    dat = dat[dat.obs["percent.mt"] < 5, :]  # 10545 × 17841

    # normalize by total UMI count: scale factor (in-place change), CPM if target=1e6
    sc.pp.normalize_total(dat, target_sum=1e4)

    # mean count for given cell type within individual
    dat.bulk = dc.get_pseudobulk(
        dat,
        sample_col="donor_id",
        groups_col="cell_type",
        mode="mean",
        min_prop=0.2,
        min_cells=0,
        min_counts=0,
        min_smpls=2,
    )

    # subset to one cell type
    dat_onetype = dat.bulk[dat.bulk.obs["cell_type"] == cell_type]
    # filter out genes with zeros reads
    dat_onetype = dat_onetype[:, np.sum(dat_onetype.X, axis=0) > 0]

    return dat_onetype


def read_data(
    geno_path: str, pheno_path: str, cell_type: str = "CD14-positive monocyte"
):
    """
    Genotype data: plink file
    pheno_path: h5ad file path, including covariates

    Gene expression data: h5ad file
    - dat.X: cell x gene sparse matrix, where cell is indexed by unique barcode
    - dat.obs: cell x features (eg. donor_id, age,...)
    - dat.var: gene x gene summary stats

    recode sex as: female = 1, male = 0
    """
    # Append prefix with suffix
    bed_path = geno_path + ".bed"
    bim_path = geno_path + ".bim"
    fam_path = geno_path + ".fam"

    # a0=0, a1=1, genotype value (0/1/2) is the count for a1 allele
    # print(G.a0.sel(variant="variant0").values)
    # print(G.sel(sample="1", variant="variant0").values)
    G = read_plink1_bin(bed_path, bim_path, fam_path, verbose=False)
    genotype = jnp.array(G.values)  # sample x variants

    dat = sc.read_h5ad(pheno_path)
    dat.obs["sex"] = np.where(dat.obs["sex"] == "female", 1, 0)
    count = process_count(dat, cell_type=cell_type)  # count and covariates

    return CleanDataState(genotype, count)
