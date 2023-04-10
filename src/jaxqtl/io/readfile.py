from typing import NamedTuple

import pandas as pd

import jax.numpy as jnp
from jax import Array

from jaxqtl.io.expr import ExpressionData, GeneMetaData
from jaxqtl.io.geno import GenoIO, PlinkReader
from jaxqtl.io.pheno import PheBedReader, PhenoIO, SingleCellFilter  # , H5AD

pd.set_option("display.max_rows", 100000)


class ReadyDataState(NamedTuple):
    geno: Array
    bim: pd.DataFrame
    pheno: ExpressionData
    gene_meta: GeneMetaData
    covar: Array


class AllDataState:
    """
    count: filtered cells and genes for given cell type, contains sample features
    """

    def __init__(
        self,
        genotype: pd.DataFrame,
        bim: pd.DataFrame,
        count: pd.DataFrame,
        covar: pd.DataFrame,
    ):
        self.geno = genotype  # nxp, index by sample iid, column names are variant names chr:pos:ref:alt
        self.bim = bim  # variant on rows
        self.pheno = count  # nxG for one cell type, count.var has gene names
        self.covar = covar  # nxcovar, covariates for the same individuals

    def get_celltype(self, cell_type: str = "CD14-positive monocyte"):
        # check orders of samples in count and genotype
        pheno_onetype = self.pheno[
            self.pheno.index.get_level_values("cell_type") == cell_type
        ]
        # drop genes with all zero expressions
        pheno_onetype = pheno_onetype.loc[:, (pheno_onetype != 0).any(axis=0)]
        gene_list = pheno_onetype.columns.values

        sample_id = pheno_onetype.index.get_level_values("donor_id").to_list()
        # filter genotype and covariates
        genotype = self.geno.loc[self.geno.index.isin(sample_id)].sort_index(
            level=sample_id
        )
        covar = self.covar.loc[self.covar.index.isin(sample_id)].sort_index(
            level=sample_id
        )

        return ReadyDataState(
            geno=jnp.float64(genotype),
            bim=self.bim,
            pheno=ExpressionData(pheno_onetype),
            gene_meta=GeneMetaData(gene_list),
            covar=jnp.float64(covar),
        )

    def format_readydata(self):
        # check order
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
            gene_meta=GeneMetaData(pos_df),
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
    pheno_path: h5ad file path, including covariates
    covar_path: covariates, must be coded in numerical forms

    Gene expression data: h5ad file
    - dat.X: cell x gene sparse matrix, where cell is indexed by unique barcode
    - dat.obs: cell x features (eg. donor_id, age,...)
    - dat.var: gene x gene summary stats

    recode sex as: female = 1, male = 0
    """
    # unfiltered genotype data
    genotype, var_info, sample_info = geno_reader(geno_path)

    covar = pd.read_csv(covar_path, delimiter="\t")
    covar = covar.set_index("iid")

    rawdat = pheno_reader(pheno_path)
    count = pheno_reader.process(rawdat, filter_opt)

    # return all data in dataframe (easier for next filtering/merging)
    return AllDataState(genotype, var_info, count, covar)
