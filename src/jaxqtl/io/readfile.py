from typing import NamedTuple

import pandas as pd

import jax.numpy as jnp
from jax import Array

from jaxqtl.io.expr import ExpressionData, GeneMetaData
from jaxqtl.io.geno import GenoIO, PlinkReader
from jaxqtl.io.pheno import PheBedReader, PhenoIO, SingleCellFilter  # , H5AD

pd.set_option("display.max_rows", 100000)


class ReadyDataState(NamedTuple):
    geno: Array  # sample x genes
    bim: pd.DataFrame
    pheno: ExpressionData
    pheno_meta: GeneMetaData
    covar: Array  # sample x covariate


class AllDataState:
    """Raw data state in data frame"""

    def __init__(
        self,
        geno: pd.DataFrame,
        bim: pd.DataFrame,
        pheno: pd.DataFrame,
        covar: pd.DataFrame,
    ):
        self.geno = (
            geno  # nxp, index by sample iid, column names are variants chr:pos:ref:alt
        )
        self.bim = bim  # variant on rows
        self.pheno = pheno  # nxG
        self.covar = covar  # nxcovar

    def create_ReadyData(self) -> ReadyDataState:
        pos_df = self.pheno[["chr", "start", "end"]].reset_index()
        self.pheno.drop(["chr", "start", "end"], axis=1, inplace=True)

        # transpose to sample x genes
        count = self.pheno.T
        sample_id = count.index.to_list()

        # filter genotype and covariates
        genotype = self.geno.loc[self.geno.index.isin(sample_id)].sort_index(
            level=sample_id
        )
        covar = self.covar.loc[self.covar.index.isin(sample_id)].sort_index(
            level=sample_id
        )

        # ensure sample order in genotype and covar are same as count
        assert (
            genotype.index == count.index
        ).all(), "samples are not sorted in genotype and count matrix"

        assert (
            covar.index == count.index
        ).all(), "samples are not sorted in covariate and count matrix"

        return ReadyDataState(
            geno=jnp.float64(genotype),
            bim=self.bim,
            pheno=ExpressionData(count),
            pheno_meta=GeneMetaData(pos_df),
            covar=jnp.float64(covar),
        )


def read_data(
    geno_path: str,
    pheno_path: str,
    covar_path: str,
    filter_opt=SingleCellFilter,
    geno_reader: GenoIO = PlinkReader(),
    pheno_reader: PhenoIO = PheBedReader(),
) -> AllDataState:
    """Read genotype, phenotype and covariates, including interaction terms
    Genotype data: plink triplet, vcf
    pheno_path: bed file
    covar_path: covariates, must be coded in numerical forms

    Gene expression data: h5ad file
    - dat.X: cell x gene sparse matrix, where cell is indexed by unique barcode
    - dat.obs: cell x features (eg. donor_id, age,...)
    - dat.var: gene x gene summary stats

    recode sex as: female = 1, male = 0
    """
    # raw genotype data
    genotype, var_info, sample_info = geno_reader(geno_path)

    covar = pd.read_csv(covar_path, delimiter="\t")
    covar = covar.set_index("iid")

    rawdat = pheno_reader(pheno_path)

    # TODO: use this for filtering
    count = pheno_reader.process(rawdat, filter_opt)

    return AllDataState(genotype, var_info, count, covar)
