from typing import NamedTuple, Tuple

import equinox as eqx
import pandas as pd

import jax.lax as lax
import jax.numpy as jnp
from jax import Array
from jax.numpy.linalg import multi_dot
from jax.scipy.stats import norm
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.glm import GLM, GLMState
from jaxqtl.io.readfile import ReadyDataState


class CisGLMState(NamedTuple):
    af: Array
    ma_count: Array
    beta: Array
    se: Array
    p: Array
    num_iters: Array
    converged: Array


class CisGLMScoreState(NamedTuple):
    p: Array
    Z: Array
    num_iters: Array
    converged: Array


def _cis_window_cutter(
    dat: ReadyDataState, chrom: str, start: int, end: int
) -> Tuple[Array, pd.DataFrame]:
    """
    return variant list in cis for given gene
    Map is a pandas data frame

    plink bim file is 1-based
    the map file is hg19,
    emsemble use 1-based
    vcf file is one-based

    gene_name = 'ENSG00000250479', start: 24110630
    GenomicRanges example: https://biocpy.github.io/GenomicRanges/

    Returns:
        Genotype matrix for cis-variants
    """
    var_info = dat.bim

    cis_var_info = var_info.loc[
        (var_info["chrom"] == str(chrom))
        & (var_info["pos"] >= start)
        & (var_info["pos"] <= end)
    ]

    # subset G to cis variants (nxp)
    G_tocheck = jnp.take(dat.geno, jnp.array(cis_var_info.i), axis=1)

    # check monomorphic: G.T[:, [0]] find first occurrence on all genotype, var x 1
    mono_var = (G_tocheck.T == G_tocheck.T[:, [0]]).all(
        1
    )  # bool (var, ), show whether given variant is monomorphic
    not_mono_var = jnp.invert(mono_var)  # reverse False and True (same as "~" operator)
    G = G_tocheck[:, not_mono_var]  # take genotype that are NOT monomorphic
    cis_var_info = cis_var_info.loc[not_mono_var.tolist()]

    # note: if no variants taken, then G has shape (n,0), cis_var_info has shape (0, 7); both 2-dim
    return G, cis_var_info


def _setup_G_y(
    dat: ReadyDataState, gene_name: str, chrom: str, start: int, end: int
) -> Tuple[Array, Array, pd.DataFrame]:
    G, var_df = _cis_window_cutter(dat, chrom, start, end)
    y = dat.pheno[gene_name]  # __getitem__

    return G, jnp.array(y), var_df


@eqx.filter_jit
def cis_scan(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    offset_eta: ArrayLike = 0.0,
    robust_se: bool = True,
    maxiter: int = 100,
) -> CisGLMState:
    """
    run GLM across variants in a flanking window of given gene
    cis-widow: plus and minus W base pairs, total length 2*cis_window
    """
    glm = GLM(family=family, max_iter=maxiter)

    # initiate SNP scan with model with covariate
    glmstate_cov_only = glm.fit(
        X, y, offset_eta=offset_eta, robust_se=robust_se, init=family.init_eta(y)
    )

    def _func(carry, snp):
        M = jnp.hstack((X, snp[:, jnp.newaxis]))
        glmstate = glm.fit(
            M,
            y,
            offset_eta=offset_eta,
            robust_se=robust_se,
            init=glmstate_cov_only.eta,
        )

        af = jnp.mean(snp) / 2.0
        snp = jnp.round(jnp.where(af <= 0.5, snp, 2 - snp))
        ma_count = jnp.sum(snp)  # Number of minor alleles

        return carry, CisGLMState(
            af=af,
            ma_count=ma_count,
            beta=glmstate.beta[-1],
            se=glmstate.se[-1],
            p=glmstate.p[-1],
            num_iters=glmstate.num_iters,
            converged=glmstate.converged,
        )

    _, state = lax.scan(_func, 0.0, G.T)

    return state


@eqx.filter_jit
def cis_scan_score(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    offset_eta: ArrayLike = 0.0,
    max_iter: int = 100,
) -> CisGLMScoreState:
    """
    run GLM across variants in a flanking window of given gene
    cis-widow: plus and minus W base pairs, total length 2*cis_window
    """
    glm = GLM(family=family, max_iter=max_iter)

    init_val = glm.family.init_eta(y)

    # fit covariate-only model (null)
    glmstate_cov_only = glm.fit(X, y, offset_eta=offset_eta, init=init_val)

    Z, pval = score_test_snp(G, X, glmstate_cov_only)

    return CisGLMScoreState(
        p=pval,
        Z=Z,
        num_iters=glmstate_cov_only.num_iters,
        converged=glmstate_cov_only.converged,
    )


def score_test_snp(
    G: ArrayLike, X: ArrayLike, glm_null_res: GLMState
) -> Tuple[Array, Array]:
    """test for additional covariate g
    only require fit null model using fitted covariate only model + new vector g
    X is the full design matrix containing covariates and g
    calculate score in full model using the model fitted from null model
    """
    y_resid = jnp.squeeze(glm_null_res.resid, -1)
    x_W = X * glm_null_res.glm_wt
    sqrt_wgt = jnp.sqrt(glm_null_res.glm_wt)

    g_resid = G - multi_dot([X, glm_null_res.infor_inv, x_W.T, G])
    w_g_resid = g_resid * sqrt_wgt
    g_var = jnp.sum(w_g_resid ** 2, axis=0)

    # TODO: SPA test; now using normal approximation
    Z = (g_resid.T @ y_resid) / jnp.sqrt(g_var)

    pval = norm.cdf(-abs(Z)) * 2

    return Z, pval
