from typing import NamedTuple, Tuple

import pandas as pd

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

    # Unit test: only test for first 10 cis variants
    # G_tocheck = jnp.take(dat.geno, jnp.array(cis_var_info.i[0:100]), axis=1)
    G_tocheck = jnp.take(dat.geno, jnp.array(cis_var_info.i), axis=1)

    # check monomorphic: G.T[:, [0]] find first occurrence on all genotype, var x 1
    mono_var = (G_tocheck.T == G_tocheck.T[:, [0]]).all(
        1
    )  # bool (var, ), show whether given variant is monomorphic
    not_mono_var = jnp.invert(mono_var)
    G = G_tocheck.T[not_mono_var].T  # take genotype that are NOT monomorphic
    cis_var_info = cis_var_info.loc[not_mono_var.tolist()]
    return G, cis_var_info


def _setup_G_y(
    dat: ReadyDataState, gene_name: str, chrom: str, start: int, end: int
) -> Tuple[Array, Array, pd.DataFrame]:
    G, var_df = _cis_window_cutter(dat, chrom, start, end)
    y = dat.pheno[gene_name]  # __getitem__

    return G, jnp.array(y), var_df


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
