from typing import List, NamedTuple

import jax.random as rdm
from jax import Array, numpy as jnp
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.permutation import BetaPerm, Permutation
from jaxqtl.infer.utils import CisGLMState, _setup_G_y, cis_scan
from jaxqtl.io.readfile import ReadyDataState


class MapCis_SingleState(NamedTuple):
    cisglm: CisGLMState
    adj_p: Array
    beta_res: Array


class MapCis_OutState(NamedTuple):
    effect_beta: List
    beta_se: List
    nominal_p: List
    adj_p: List
    beta_param: List
    converged: List


# write this assuming this is bulk data
def map_cis(
    dat: ReadyDataState,
    family: ExponentialFamily,
    seed: int = 123,
    window: int = 500000,
    sig_level: float = 0.05,
    perm: Permutation = BetaPerm(),
) -> MapCis_OutState:
    n, k = dat.covar.shape
    gene_info = dat.pheno_meta

    # append genotype as the last column
    X = jnp.hstack((jnp.ones((n, 1)), dat.covar))
    key = rdm.PRNGKey(seed)

    effect_beta = []
    beta_se = []
    nominal_p = []
    adj_p = []
    beta_param = []
    converged = []

    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = min(0, start_min - window)
        rend = end_max + window

        # pull cis G and y for this gene
        G, y = _setup_G_y(dat, gene_name, str(chrom), lstart, rend)

        # skip if no cis SNPs found
        if G.shape[1] == 0:
            continue

        key, g_key = rdm.split(key)

        result = map_cis_single(
            X,
            G,
            y,
            family,
            g_key,
            sig_level,
            perm,
        )

        # combine results
        effect_beta.append(result.cisglm.beta)
        beta_se.append(result.cisglm.se)
        nominal_p.append(result.cisglm.p)
        adj_p.append(result.adj_p)
        beta_param.append(result.beta_res)
        converged.append(result.cisglm.converged)

        # unit test for 4 genes
        if len(adj_p) > 3:
            break

    # filter results based on user speicification (e.g., report all, report top, etc)

    return MapCis_OutState(
        effect_beta=effect_beta,
        beta_se=beta_se,
        nominal_p=nominal_p,
        adj_p=adj_p,
        beta_param=beta_param,
        converged=converged,
    )


def map_cis_single(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    key_init,
    sig_level=0.05,
    perm: Permutation = BetaPerm(),
) -> MapCis_SingleState:
    """Generate result of GLM for variants in cis
    For given gene, find all variants in + and - window size TSS region

    window: width of flanking on either side of TSS
    sig_level: desired significance level (not used)
    perm: Permutation method
    """

    cisglmstate = cis_scan(X, G, y, family)

    adj_p, beta_res = perm(
        X,
        y,
        G,
        jnp.min(cisglmstate.p),
        family,
        key_init,
        sig_level,
    )

    return MapCis_SingleState(cisglm=cisglmstate, adj_p=adj_p, beta_res=beta_res)


def map_cis_nominal(
    dat: ReadyDataState,
    family: ExponentialFamily,
    seed: int = 123,
    window: int = 500000,
) -> MapCis_OutState:
    n, k = dat.covar.shape
    gene_info = dat.pheno_meta

    # append genotype as the last column
    X = jnp.hstack((jnp.ones((n, 1)), dat.covar))
    key = rdm.PRNGKey(seed)

    effect_beta = []
    beta_se = []
    nominal_p = []

    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = min(0, start_min - window)
        rend = end_max + window

        # pull cis G and y for this gene
        G, y = _setup_G_y(dat, gene_name, str(chrom), lstart, rend)

        # skip if no cis SNPs found
        if G.shape[1] == 0:
            continue

        key, g_key = rdm.split(key)

        result = cis_scan(X, G, y, family)

        # combine results
        effect_beta.append(result.beta)
        beta_se.append(result.se)
        nominal_p.append(result.p)

        # unit test for 4 genes
        if len(nominal_p) > 3:
            break

    # filter results based on user speicification (e.g., report all, report top, etc)

    return MapCis_OutState(
        effect_beta=effect_beta,
        beta_se=beta_se,
        nominal_p=nominal_p,
        adj_p=[],
        beta_param=[],
        converged=[],
    )
