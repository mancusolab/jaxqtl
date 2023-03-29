from typing import Optional

from jax import numpy as jnp

from jaxqtl.families.distribution import ExponentialFamily, Poisson
from jaxqtl.infer.glm import GLM, GLMState
from jaxqtl.io.readfile import CleanDataState


def run_bigGLM(
    dat: CleanDataState,
    family: ExponentialFamily = Poisson(),
    test_run: Optional[int] = None,
):
    G = dat.genotype  # n x p variants
    covar = dat.covar
    nobs, pvar = G.shape

    num_params = covar.shape[1] + 2  # covariate features + one SNP + intercept
    num_genes = test_run if test_run is not None else dat.count.X.shape[1]

    # num_var = 1000  # G.shape[1]
    all_beta = jnp.zeros((num_params, num_genes))
    all_se = jnp.zeros((num_params, num_genes))
    all_pval = jnp.zeros((num_params, num_genes))
    all_num_iters = jnp.zeros((num_genes,))
    all_converged = jnp.zeros((num_genes,))

    # Xmat: intercept, SNP, cov1, cov2, ...
    Xmat = jnp.ones((nobs, num_params))
    Xmat = Xmat.at[:, 2:].set(covar)

    for idx in range(num_genes):
        Xmat = Xmat.at[:, 1].set(G[:, idx])  # append X with genotype
        ycount = dat.count.X[:, idx].astype("float64")

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
