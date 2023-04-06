from typing import NamedTuple

import pandas as pd
from anndata._core.anndata import AnnData

from jax import Array

from jaxqtl.io.geno import GenoIO
from jaxqtl.io.pheno import PhenoIO, SingleCellFilter

pd.set_option("display.max_rows", 100000)


class CleanDataState(NamedTuple):
    """
    count: filtered cells and genes for given cell type, contains sample features
    """

    genotype: Array  # nxp, index by sample iid, column names are variant names chr:pos:ref:alt
    bim: pd.DataFrame  # variant on rows
    count: AnnData  # nxG for one cell type, count.var has gene names
    covar: pd.DataFrame  # nxcovar, covariates for the same individuals


def read_data(
    geno_path: str,
    pheno_path: str,
    covar_path: str,
    filter_opt: SingleCellFilter,
    geno_reader: GenoIO,
    pheno_reader: PhenoIO,
) -> CleanDataState:
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
    genotype, var_info, sample_info = geno_reader(geno_path)
    covar = pd.read_csv(covar_path, delimiter="\t")
    covar = covar.set_index("donor_id")

    rawdat = pheno_reader(pheno_path)
    count = rawdat.process(filter_opt)
    donor_id = count.obs.donor_id.values

    # filter genotype and covariates, ordered?
    genotype = genotype.filter(items=donor_id, axis=0)
    covar = covar.filter(items=donor_id, axis=0)

    return CleanDataState(genotype, var_info, count, covar)
