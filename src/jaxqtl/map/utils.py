from typing import NamedTuple, Optional, Tuple

import pandas as pd

import jax.scipy as sp

from jax import numpy as jnp
from jax._src.basearray import ArrayLike
from jaxtyping import Array

from jaxqtl.io.readfile import ReadyDataState


class _GenoInfo(NamedTuple):
    af: Array
    ma_count: Array


def _get_geno_info(G: ArrayLike) -> _GenoInfo:
    n, p = G.shape
    counts = jnp.sum(G, axis=0)  # count REF allele
    af = counts / (2.0 * n)
    flag = af <= 0.5
    ma_counts = jnp.where(flag, counts, 2 * n - counts)

    return _GenoInfo(af, ma_counts)


def _cis_window_cutter(dat: ReadyDataState, chrom: str, start: int, end: int) -> Tuple[Array, pd.DataFrame]:
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
        (var_info["chrom"] == str(chrom)) & (var_info["pos"] >= start) & (var_info["pos"] <= end)
    ]

    # subset G to cis variants (nxp)
    G_tocheck = jnp.take(dat.geno, jnp.array(cis_var_info.i), axis=1)

    # check monomorphic: G.T[:, [0]] find first occurrence on all genotype, var x 1
    mono_var = (G_tocheck.T == G_tocheck.T[:, [0]]).all(1)  # bool (var, ), show whether given variant is monomorphic
    not_mono_var = jnp.invert(mono_var)  # reverse False and True (same as "~" operator)
    G = G_tocheck[:, not_mono_var]  # take genotype that are NOT monomorphic
    cis_var_info = cis_var_info.loc[not_mono_var.tolist()]

    # note: if no variants taken, then G has shape (n,0), cis_var_info has shape (0, 7); both 2-dim
    return G, cis_var_info


def _check_geno(dat: ReadyDataState) -> Tuple[Array, pd.DataFrame]:
    cis_var_info = dat.bim

    # subset G to cis variants (nxp)
    G_tocheck = dat.geno

    # check monomorphic: G.T[:, [0]] find first occurrence on all genotype, var x 1
    mono_var = (G_tocheck.T == G_tocheck.T[:, [0]]).all(1)  # bool (var, ), show whether given variant is monomorphic
    not_mono_var = jnp.invert(mono_var)  # reverse False and True (same as "~" operator)
    G = G_tocheck[:, not_mono_var]  # take genotype that are NOT monomorphic
    cis_var_info = cis_var_info.loc[not_mono_var.tolist()]

    # note: if no variants taken, then G has shape (n,0), cis_var_info has shape (0, 7); both 2-dim
    return G, cis_var_info


def _setup_G_y(
    dat: ReadyDataState, gene_name: str, chrom: str, start: int, end: int, mode: Optional[str] = None
) -> Tuple[Array, Array, pd.DataFrame]:
    if mode == "trans":
        G, var_df = _check_geno(dat)
    else:
        G, var_df = _cis_window_cutter(dat, chrom, start, end)

    y = dat.pheno[gene_name]  # __getitem__

    return G, jnp.array(y), var_df


def _ACAT(pvalues: ArrayLike, weights: Optional[ArrayLike] = None) -> Array:
    '''acat_test()
    # ref: https://gist.github.com/ryananeff/c66cdf086979b13e855f2c3d0f3e54e1
    Aggregated Cauchy Assocaition Test
    A p-value combination method using the Cauchy distribution.

    Inspired by: https://github.com/yaowuliu/ACAT/blob/master/R/ACAT.R

    Author: Ryan Neff

    Inputs:
        pvalues: <list or numpy array>
            The p-values you want to combine.
        weights: <list or numpy array>, default=None
            The weights for each of the p-values. If None, equal weights are used.

    Returns:
        pval: <float>
            The ACAT combined p-value.
    '''
    if any(jnp.isnan(pvalues)):
        raise Exception("Cannot have NAs in the p-values.")
    if any([(i > 1) | (i < 0) for i in pvalues]):
        raise Exception("P-values must be between 0 and 1.")
    if any([i == 1 for i in pvalues]) & any([i == 0 for i in pvalues]):
        raise Exception("Cannot have both 0 and 1 p-values.")
    if any([i == 0 for i in pvalues]):
        print("Warn: p-values are exactly 0.")
        return jnp.array([0])
    if any([i == 1 for i in pvalues]):
        print("Warn: p-values are exactly 1.")
        return jnp.array([1])
    if weights is None:
        weights = [1 / len(pvalues) for i in pvalues]
    elif len(weights) != len(pvalues):
        raise Exception("Length of weights and p-values differs.")
    elif any([i < 0 for i in weights]):
        raise Exception("All weights must be positive.")
    else:
        weights = [i / len(weights) for i in weights]

    pvalues = jnp.array(pvalues)
    weights = jnp.array(weights)

    if any([i < 1e-16 for i in pvalues]) is False:
        cct_stat = sum(weights * jnp.tan((0.5 - pvalues) * jnp.pi))
    else:
        is_small = [i < (1e-16) for i in pvalues]
        is_large = [i >= (1e-16) for i in pvalues]
        cct_stat = sum((weights[is_small] / pvalues[is_small]) / jnp.pi)
        cct_stat += sum(weights[is_large] * jnp.tan((0.5 - pvalues[is_large]) * jnp.pi))

    if cct_stat > 1e15:
        pval = (1 / cct_stat) / jnp.pi
    else:
        pval = 1 - sp.stats.cauchy.cdf(cct_stat)

    return pval
