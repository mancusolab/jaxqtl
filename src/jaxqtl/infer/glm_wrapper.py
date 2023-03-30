# from typing import Optional

from jax import numpy as jnp

from jaxqtl.families.distribution import ExponentialFamily, Poisson
from jaxqtl.infer.glm import GLM, GLMState
from jaxqtl.io.readfile import CleanDataState

# from .utils import cis_window_cutter


def run_cis_GLM(
    dat: CleanDataState, family: ExponentialFamily = Poisson(), gene_idx=0, W=1e6
):
    """
    run GLM across variants in a flanking window of given gene
    cis-widow: plus and minus W base pairs, total length 2*W
    """
    G = dat.genotype  # n x p variants
    covar = dat.covar
    nobs, pvar = G.shape

    num_params = covar.shape[1] + 2  # covariate features + one SNP + intercept

    cis_window = 10

    all_beta = jnp.zeros((num_params, cis_window))
    all_se = jnp.zeros((num_params, cis_window))
    all_pval = jnp.zeros((num_params, cis_window))
    all_num_iters = jnp.zeros((cis_window,))
    all_converged = jnp.zeros((cis_window,))

    # Xmat: intercept, SNP, cov1, cov2, ...
    Xmat = jnp.ones((nobs, num_params))
    Xmat = Xmat.at[:, 2:].set(covar)

    for idx in range(cis_window):
        Xmat = Xmat.at[:, 1].set(G[:, idx])  # append X with genotype
        ycount = dat.count.X[:, gene_idx].astype("float64")

        glmstate = GLM(
            X=Xmat,
            y=ycount,
            family=family,
            append=False,
            maxiter=1000,
        ).fit()

        all_beta = all_beta.at[:, idx].set(glmstate.beta)
        all_se = all_se.at[:, idx].set(glmstate.se)
        all_pval = all_pval.at[:, idx].set(glmstate.p)
        all_num_iters = all_num_iters.at[idx].set(glmstate.num_iters)
        all_converged = all_converged.at[idx].set(glmstate.converged)

    return GLMState(all_beta, all_se, all_pval, all_num_iters, all_converged)
