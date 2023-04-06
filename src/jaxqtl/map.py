import jax.random as rdm
from jax import numpy as jnp
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.permutation import BetaPerm, Permutation
from jaxqtl.infer.utils import _setup_G_y, cis_scan
from jaxqtl.io.expr import GeneMetaData
from jaxqtl.io.readfile import CleanDataState


def map_cis(
    dat: CleanDataState,
    gene_info: GeneMetaData,
    family: ExponentialFamily,
    seed: int,
    window: int = 500000,
    sig_level: float = 0.05,
    perm: Permutation = BetaPerm(),
):
    n, k = dat.covar.shape

    # append genotype as the last column
    X = jnp.hstack((jnp.ones(n, 1), dat.covar))
    key = rdm.PRNGKey(seed)

    for gene in gene_info:
        name, chrom, start_min, end_max = gene
        lstart = min(0, start_min - window)
        rend = end_max + window
        G, y = _setup_G_y(dat, name, chrom, lstart, rend)
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
        print(result)

    # combine results somehow
    # return results

    pass


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

    return cisglmstate, adj_p, beta_res[0], beta_res[1]
