from typing import List, NamedTuple

import jax.random as rdm
from jax import Array, numpy as jnp
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.permutation import BetaPerm, Permutation
from jaxqtl.infer.utils import CisGLMState, _setup_G_y, cis_scan
from jaxqtl.io.expr import GeneMetaData
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


# write this assuming this is bulk data
def map_cis(
    dat: ReadyDataState,
    gene_info: GeneMetaData,
    family: ExponentialFamily,
    seed: int = 123,
    window: int = 500000,
    sig_level: float = 0.05,
    perm: Permutation = BetaPerm(),
):
    n, k = dat.covar.shape

    # append genotype as the last column
    X = jnp.hstack((jnp.ones((n, 1)), dat.covar))
    key = rdm.PRNGKey(seed)

    effect_beta = []
    beta_se = []
    nominal_p = []
    adj_p = []
    beta_param = []

    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = min(0, start_min - window)
        rend = end_max + window

        # pull cis G and y for this gene
        G, y = _setup_G_y(dat, gene_name, str(chrom), lstart, rend)
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
        # filter results based on user speicification (e.g., report all, report top, etc)

        # combine results
        effect_beta.append(result.cisglm.beta)
        beta_se.append(result.cisglm.se)
        nominal_p.append(result.cisglm.p)
        adj_p.append(result.adj_p)
        beta_param.append(result.beta_res)

        if len(adj_p) > 3:
            break

    return MapCis_OutState(
        effect_beta=effect_beta,
        beta_se=beta_se,
        nominal_p=nominal_p,
        adj_p=adj_p,
        beta_param=beta_param,
    )


def map_cis_single(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    key_init,
    sig_level=0.05,
    perm: Permutation = BetaPerm(),
):
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
