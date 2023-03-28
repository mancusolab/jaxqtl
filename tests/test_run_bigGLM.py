import statsmodels.api as sm
from statsmodels.discrete.discrete_model import (  # ,NegativeBinomial
    Poisson as smPoisson,
)
from util import assert_beta_array_eq

import jax.numpy as jnp
from jax.config import config

from jaxqtl.infer.glm import GLMState, run_bigGLM
from jaxqtl.io.readfile import read_data

config.update("jax_enable_x64", True)

geno_path = "./tests/data/onek1k"
pheno_path = "./tests/data/Countdata.h5ad"
# pheno_path = "../NextProject/data/OneK1K/Count.h5ad"

cell_type = "CD14-positive monocyte"
dat = read_data(geno_path, pheno_path, cell_type)


def run_bigGLM_sm(dat, test_run):
    # TODO: order of genotype is not same as count matrix
    # TODO: use donor_id as family id when creating plink file
    # G = dat.genotype  # n x p variants
    Xmat = dat.count.obs[["sex", "age"]].astype("float64")
    Xmat = sm.add_constant(Xmat, prepend=True)  # X

    num_params = Xmat.shape[1]  # features + intercept
    num_genes = test_run if test_run is not None else dat.count.X.shape[1]

    # num_var = 1000  # G.shape[1]
    all_beta = jnp.zeros((num_params, num_genes))
    all_se = jnp.zeros((num_params, num_genes))
    all_pval = jnp.zeros((num_params, num_genes))
    all_num_iters = jnp.zeros((num_genes,))
    all_converged = jnp.zeros((num_genes,))

    for idx in range(num_genes):
        # Xmat["variant"] = G[:, idx] # append X with genotype
        ycount = dat.count.X[:, idx].astype("float64")
        glmstate = smPoisson(ycount, Xmat).fit(disp=0, full_output=True)

        all_beta[:, idx] = glmstate.params
        all_se[:, idx] = glmstate.bse
        all_pval[:, idx] = glmstate.pvalues
        # all_num_iters[idx] = glmstate.num_iters
        # all_converged[idx] = glmstate.converged

    return GLMState(all_beta, all_se, all_pval, all_num_iters, all_converged)


def test_run_bigGLM():
    # 940 samples x 12733 genes
    smstate = run_bigGLM_sm(dat, test_run=10)
    glmstate = run_bigGLM(dat, test_run=100)
    assert_beta_array_eq(glmstate, smstate)
