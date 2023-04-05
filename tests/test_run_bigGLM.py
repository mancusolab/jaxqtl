from typing import NamedTuple

import numpy as np
from statsmodels.discrete.discrete_model import (  # ,NegativeBinomial
    Poisson as smPoisson,
)

from jax import random
from jax.config import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.infer.glm_wrapper import map_cis
from jaxqtl.infer.utils import _setup_X_y, cis_window_cutter
from jaxqtl.io.readfile import CYVCF2, CleanDataState, read_data

# from utils import assert_beta_array_eq


config.update("jax_enable_x64", True)


class smState(NamedTuple):
    beta: np.array
    se: np.array
    p: np.array


geno_path = "./example/data/chr22"
pheno_path = "./example/data/Countdata_n100.h5ad"
covar_path = "./example/data/donor_features.tsv"
# pheno_path = "../NextProject/data/OneK1K/Count.h5ad"

cell_type = "CD14-positive monocyte"
dat = read_data(CYVCF2(), geno_path, pheno_path, covar_path, cell_type)
key = random.PRNGKey(1)
key, key_init = random.split(key, 2)


# TODO: need error handle singlular value (won't stop for now, but Inf estimate in SE)
glmstate, p, k, n = map_cis(
    dat,
    family=Poisson(),
    gene_name="ENSG00000250479",
    key_init=key_init,
    max_perm_direct=100,
)


def run_cis_GLM_sm(dat: CleanDataState, gene_name: str, window: int = 1000000):

    X, y = _setup_X_y(dat, gene_name)
    G = dat.genotype
    y = np.array(y)
    cis_list = cis_window_cutter(G, gene_name, dat.var_info, window)
    cis_num = len(cis_list)

    all_beta = np.zeros((cis_num,))
    all_se = np.zeros((cis_num,))
    all_pval = np.zeros((cis_num,))

    for idx in range(len(cis_list)):
        print(idx)
        X = X.at[:, 1].set(G[cis_list[idx]])  # append X with genotype
        smstate = smPoisson(y, np.array(X)).fit(disp=0, full_output=True)

        all_beta[idx] = smstate.params[1]
        all_se[idx] = smstate.bse[1]
        all_pval[idx] = smstate.pvalues[1]

    return smState(all_beta, all_se, all_pval)


# TODO: not sure why some estimates are very different
smstate = run_cis_GLM_sm(dat, gene_name="ENSG00000250479")


# def test_run_cis_GLM():
#     # 940 samples x 12733 genes
#     smstate = run_cis_GLM_sm(dat)
#     glmstate = run_cis_GLM(dat, family=Poisson())
#     assert_beta_array_eq(glmstate, smstate)
