from typing import Optional

from jax import numpy as jnp

from jaxqtl.families.distribution import ExponentialFamily, Poisson
from jaxqtl.infer.glm import GLM, GLMState


def run_bigGLM(
    dat, family: ExponentialFamily = Poisson(), test_run: Optional[int] = None
):
    # TODO: order of genotype is not same as count matrix
    # TODO: use donor_id as family id when creating plink file
    # G = dat.genotype  # n x p variants
    Xmat = dat.count.obs[["sex", "age"]].astype("float64")
    num_params = Xmat.shape[1] + 1  # features + intercept
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

        glmstate = GLM(
            X=Xmat,
            y=ycount,
            family=family,
            append=True,
            maxiter=1000,
        ).fit()

        all_beta[:, idx] = glmstate.beta
        all_se[:, idx] = glmstate.se
        all_pval[:, idx] = glmstate.p
        all_num_iters[idx] = glmstate.num_iters
        all_converged[idx] = glmstate.converged

    return GLMState(all_beta, all_se, all_pval, all_num_iters, all_converged)
