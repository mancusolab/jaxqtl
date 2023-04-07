from typing import NamedTuple, Tuple

import jax.lax as lax
from jax import Array, numpy as jnp
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.glm import GLM
from jaxqtl.io.readfile import ReadyDataState


class CisGLMState(NamedTuple):
    beta: jnp.ndarray
    se: jnp.ndarray
    p: jnp.ndarray
    num_iters: jnp.ndarray
    converged: jnp.ndarray


def _cis_window_cutter(dat: ReadyDataState, chrom: str, start: int, end: int) -> Array:
    """
    return variant list in cis for given gene
    Map is a pandas data frame

    plink bim file is 1-based
    the map file is hg19,
    emsemble use 1-based
    vcf file is one-based

    gene_name = 'ENSG00000250479'
    GenomicRanges example: https://biocpy.github.io/GenomicRanges/
    """
    var_info = dat.bim
    chrom = "22"
    cis_var_info = var_info.loc[
        (var_info["chrom"] == chrom)
        & (var_info["pos"] >= start)
        & (var_info["pos"] <= end)
    ]

    G = jnp.take(dat.geno, jnp.array(cis_var_info.i), axis=1)

    # return column indices
    return G


def _setup_G_y(
    dat: ReadyDataState, gene_name: str, chrom: str, start: int, end: int
) -> Tuple[Array, Array]:
    G = _cis_window_cutter(dat, chrom, start, end)
    y = dat.pheno[gene_name]  # __getitem__

    return G, jnp.array(y)


def cis_scan(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
) -> CisGLMState:
    """
    run GLM across variants in a flanking window of given gene
    cis-widow: plus and minus W base pairs, total length 2*cis_window
    """

    def _func(carry, snp):
        M = jnp.hstack((X, snp[:, jnp.newaxis]))
        glmstate = GLM(
            X=M,
            y=y,
            family=family,
            append=False,
            maxiter=100,
        ).fit()
        return carry, CisGLMState(
            beta=glmstate.beta[-1],
            se=glmstate.se[-1],
            p=glmstate.p[-1],
            num_iters=glmstate.num_iters,
            converged=glmstate.converged,
        )

    _, state = lax.scan(_func, 0.0, G.T)
    return state

    # beta = []
    # se = []
    # p = []
    # num_iters = []
    # converged = []
    #
    # for snp in G.T:
    #     M = jnp.hstack((X, snp[:, jnp.newaxis]))
    #     glmstate = GLM(
    #         X=M,
    #         y=y,
    #         family=family,
    #         append=False,
    #         maxiter=100,
    #     ).fit()
    #     beta.append(glmstate.beta[-1])
    #     se.append(glmstate.se[-1])
    #     p.append(glmstate.p[-1])
    #     num_iters.append(glmstate.num_iters)
    #     converged.append(glmstate.converged)
    #
    # return CisGLMState(jnp.array(beta),
    #                    jnp.array(se),
    #                    jnp.array(p),
    #                    jnp.array(num_iters),
    #                    jnp.array(converged))
