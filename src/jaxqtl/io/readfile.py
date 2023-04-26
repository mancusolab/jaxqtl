from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import jax.numpy as jnp
from jax import Array

from jaxqtl.io.covar import covar_reader
from jaxqtl.io.expr import ExpressionData, GeneMetaData
from jaxqtl.io.geno import GenoIO, PlinkReader
from jaxqtl.io.pheno import PheBedReader, PhenoIO  # , H5AD, SingleCellFilter
from jaxqtl.log import get_log


@dataclass
class ReadyDataState:
    geno: Array  # sample x genes
    bim: pd.DataFrame
    pheno: ExpressionData
    pheno_meta: GeneMetaData
    covar: Array  # sample x covariate

    def filter_geno(self, maf_threshold: float = 0.0, *chrom):
        self.pheno_meta.filter_chr(*chrom)

        # filter bim by chr and maf
        self.bim = self.bim.loc[self.bim.chrom.isin(chrom)]

        assert 0 <= maf_threshold <= 1, "maf threshold must be in range [0, 1]"
        if maf_threshold > 0.0:
            af = np.array(self.geno.mean(axis=0) / 2)
            maf = np.where(af > 0.5, 1 - af, af)  # convert to maf
            self.bim = self.bim.loc[maf > maf_threshold]

        # pull filtered geno by bim file
        self.geno = jnp.take(self.geno, jnp.array(self.bim.i), axis=1)
        assert self.geno.shape[1] == len(
            self.bim
        ), "genotype and bim file do not have same shape"

        # reset "i" for pulling genotype by position
        self.bim.i = np.arange(0, self.geno.shape[1])

    def transform_y(self, y0: float = 1.0, log_y: bool = False):
        # add dispersion shift term
        self.pheno.count = self.pheno.count + y0
        if log_y:
            self.pheno.count = jnp.log(self.pheno.count)  # prevent log(0)

    def add_covar_pheno_PC(self, k: int):
        pca_pheno = PCA(n_components=k)
        pca_res = pca_pheno.fit(self.pheno.count.T)
        PCs = jnp.array(pca_res.components_.T)  # nxk
        self.covar = jnp.hstack((self.covar, PCs))  # append k expression PCs in pheno


def read_data(
    geno_path: str,
    pheno_path: str,
    covar_path: str,
    geno_reader: GenoIO = PlinkReader(),
    pheno_reader: PhenoIO = PheBedReader(),
    log=None,
) -> ReadyDataState:
    """Read genotype, phenotype and covariates, including interaction terms
    Genotype data: plink triplet, vcf
    pheno_path: bed file
    covar_path: covariates, must be coded in numerical forms

    Gene expression data: h5ad file
    - dat.X: cell x gene sparse matrix, where cell is indexed by unique barcode
    - dat.obs: cell x features (eg. donor_id, age,...)
    - dat.var: gene x gene summary stats

    recode sex as: female = 1, male = 0

    All these files must contain the same set of individuals (better ordered, but not required)
    Internally we check ordering and guarantee they are in the same order as phenotype data
    """
    if log is None:
        log = get_log()

    # raw genotype data and impute for genotype data
    log.info("Load genotype.")
    geno, bim, sample_info = geno_reader(geno_path)

    log.info("Load covariates.")
    covar = covar_reader(covar_path)

    log.info("Load phenotype.")
    pheno = pheno_reader(pheno_path)

    # put gene name (index) back to columns
    pos_df = pheno[["chr", "start", "end"]].reset_index()
    pheno.drop(["chr", "start", "end"], axis=1, inplace=True)

    # transpose to sample x genes
    pheno = pheno.T
    pheno.columns.name = None  # remove column name due to tranpose
    sample_id = pheno.index.to_list()

    # filter genotype and covariates by sample id
    geno = geno.loc[geno.index.isin(sample_id)].sort_index(level=sample_id)
    covar = covar.loc[covar.index.isin(sample_id)].sort_index(level=sample_id)

    # ensure sample order in genotype and covar are same as count
    assert (
        geno.index == pheno.index
    ).all(), "samples are not sorted in genotype and count matrix"

    assert (
        covar.index == pheno.index
    ).all(), "samples are not sorted in covariate and count matrix"

    log.info("Finished loading raw data.")

    return ReadyDataState(
        geno=jnp.float64(geno),
        bim=bim,
        pheno=ExpressionData(pheno),
        pheno_meta=GeneMetaData(pos_df),
        covar=jnp.float64(covar),
    )
