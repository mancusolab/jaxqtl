from typing import NamedTuple

import numpy as np
from statsmodels.discrete.discrete_model import (  # ,NegativeBinomial
    Poisson as smPoisson,
)
from utils import assert_beta_array_eq

from jax.config import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.infer.glm_wrapper import run_cis_GLM
from jaxqtl.io.readfile import CYVCF2, read_data

config.update("jax_enable_x64", True)


class smState(NamedTuple):
    beta: np.array
    se: np.array
    p: np.array


geno_path = "./example/data/chr22"
pheno_path = "./example/data/Countdata.h5ad"
covar_path = "./example/data/donor_features.tsv"
# pheno_path = "../NextProject/data/OneK1K/Count.h5ad"

cell_type = "CD14-positive monocyte"
dat = read_data(CYVCF2(), geno_path, pheno_path, covar_path, cell_type)
# res = run_bigGLM(dat, family=Poisson(), test_run=10)


def run_cis_GLM_sm(dat, gene_idx=0):
    G = dat.genotype  # n x p variants
    covar = dat.covar
    nobs, pvar = G.shape

    num_params = covar.shape[1] + 2  # covariate features + one SNP + intercept

    cis_window = 10
    all_beta = np.zeros((num_params, cis_window))
    all_se = np.zeros((num_params, cis_window))
    all_pval = np.zeros((num_params, cis_window))

    # Xmat: intercept, SNP, cov1, cov2, ...
    Xmat = np.ones((nobs, num_params))
    Xmat[:, 2:] = covar

    for idx in range(cis_window):
        Xmat[:, 1] = G[:, idx]  # append X with genotype
        ycount = dat.count.X[:, gene_idx].astype("float64")

        glmstate = smPoisson(ycount, Xmat).fit(disp=0, full_output=True)

        all_beta[:, idx] = glmstate.params
        all_se[:, idx] = glmstate.bse
        all_pval[:, idx] = glmstate.pvalues

    return smState(all_beta, all_se, all_pval)


def test_run_cis_GLM():
    # 940 samples x 12733 genes
    smstate = run_cis_GLM_sm(dat)
    glmstate = run_cis_GLM(dat, family=Poisson())
    assert_beta_array_eq(glmstate, smstate)
