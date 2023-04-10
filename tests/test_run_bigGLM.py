from typing import NamedTuple

import numpy as np

# from jax import random
from jax.config import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.io.geno import PlinkReader  # , VCFReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import read_data
from jaxqtl.map import map_cis

# from statsmodels.discrete.discrete_model import (  # ,NegativeBinomial
#     Poisson as smPoisson,
# )
# from utils import assert_array_eq


config.update("jax_enable_x64", True)


class smState(NamedTuple):
    beta: np.array
    se: np.array
    p: np.array


geno_path = "./example/data/chr22.bed"
# pheno_path = "./example/data/Countdata_n100.h5ad"
covar_path = "./example/data/donor_features.tsv"
pheno_path = "./example/data/CD14_positive_monocyte.bed.gz"
# pheno_path = "../NextProject/data/OneK1K/Count.h5ad"

cell_type = "CD14-positive monocyte"
dat = read_data(
    geno_path,
    pheno_path,
    covar_path,
    cell_type,
    geno_reader=PlinkReader(),
    pheno_reader=PheBedReader(),
)  # Plink(), CYVCF2()

# filter by cell type
# dat_CD14 = dat.get_celltype(cell_type)
dat_CD14 = dat.format_readydata()

# TODO: need error handle singlular value (won't stop for now, but Inf estimate in SE)
mapcis_out = map_cis(dat_CD14, dat_CD14.gene_meta, family=Poisson(), seed=123)
print(mapcis_out.effect_beta)

# def run_cis_GLM_sm(dat: AllDataState, gene_name: str, window: int = 500000):
#
#     n, k = dat.covar.shape
#     X = np.hstack((np.ones((n, 1), dat.covar)))
#     lstart = min(0, 23765834 - window)
#     rend = 23767972 + window
#     G, y = _setup_G_y(dat, gene_name, "22", lstart, rend)
#
#     cis_num = G.shape[1]
#     all_beta = np.zeros((cis_num,))
#     all_se = np.zeros((cis_num,))
#     all_pval = np.zeros((cis_num,))
#
#     for idx, snp in enumerate(G.T):
#         M = np.hstack((X, snp[:, np.newaxis]))
#         smstate = smPoisson(y, M).fit(disp=0, full_output=True)
#
#         all_beta[idx] = smstate.params[-1]
#         all_se[idx] = smstate.bse[-1]
#         all_pval[idx] = smstate.pvalues[-1]
#
#     return smState(all_beta, all_se, all_pval)
#
#
# # TODO: not sure why some estimates are very different
# # smstate = run_cis_GLM_sm(dat, gene_name="ENSG00000250479")
#
#
# def test_run_cis_GLM():
#     # 940 samples x 12733 genes
#     # smstate = run_cis_GLM_sm(dat, gene_name="ENSG00000250479")
#     # glmstate, p, k, n = map_cis(
#     #     dat,
#     #     family=Poisson(),
#     #     gene_name="ENSG00000250479",
#     #     key_init=key_init,
#     #     max_perm_direct=0,
#     # )
#
#     assert_array_eq(glmstate.p, smstate.p)
