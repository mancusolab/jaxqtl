from dataclasses import dataclass

import numpy as np
import pandas as pd

import jax.numpy as jnp
from jax import Array

from jaxqtl.io.covar import covar_reader
from jaxqtl.io.expr import ExpressionData, GeneMetaData
from jaxqtl.io.geno import GenoIO, PlinkReader
from jaxqtl.io.pheno import PheBedReader, PhenoIO  # , H5AD, SingleCellFilter

# pd.set_option("display.max_rows", 100000)


@dataclass
class ReadyDataState:
    geno: Array  # sample x genes
    bim: pd.DataFrame
    pheno: ExpressionData
    pheno_meta: GeneMetaData
    covar: Array  # sample x covariate

    def filter_geno(self, chrom: str):
        self.pheno_meta.filter_chr(chrom)
        self.bim = self.bim.loc[self.bim.chrom == chrom]

        # pull genotype for this chrom only and reset "i" for pulling genotype by position
        self.geno = jnp.take(self.geno, jnp.array(self.bim.i), axis=1)
        self.bim.i = np.arange(0, self.geno.shape[1])


def read_data(
    geno_path: str,
    pheno_path: str,
    covar_path: str,
    maf_threshold: float = 0.0,
    geno_reader: GenoIO = PlinkReader(),
    pheno_reader: PhenoIO = PheBedReader(),
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
    # raw genotype data
    geno, bim, sample_info = geno_reader(geno_path)
    geno, bim = geno_reader.filter_geno(geno, bim, maf_threshold)

    covar = covar_reader(covar_path)

    pheno = pheno_reader(pheno_path)

    pos_df = pheno[["chr", "start", "end"]].reset_index()
    pheno.drop(["chr", "start", "end"], axis=1, inplace=True)

    # transpose to sample x genes
    pheno = pheno.T
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

    return ReadyDataState(
        geno=jnp.float64(geno),
        bim=bim,
        pheno=ExpressionData(pheno),
        pheno_meta=GeneMetaData(pos_df),
        covar=jnp.float64(covar),
    )
