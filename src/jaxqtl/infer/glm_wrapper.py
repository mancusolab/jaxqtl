from jax import numpy as jnp

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.permutation import BetaPerm, Permutation
from jaxqtl.infer.utils import _cis_window_cutter, _setup_X_y, cis_scan
from jaxqtl.io.readfile import CleanDataState


def map_cis(
    dat: CleanDataState,
    family: ExponentialFamily,
    gene_name: str,
    key_init,
    window: int = 1000000,
    sig_level=0.05,
    perm: Permutation = BetaPerm(),
    max_perm_direct=100,
    max_perm_beta=100,
):
    """Generate result of GLM for variants in cis
    For given gene, find all variants in + and - window size TSS region

    window: width of flanking on either side of TSS
    sig_level: desired significance level (not used)
    perm: Permutation method
    """
    # convert df to jnp.array
    X, y = _setup_X_y(dat, gene_name)
    G = jnp.asarray(dat.genotype)
    bim = dat.bim.ID.to_list()

    cis_list = _cis_window_cutter(gene_name, dat.bim, window)
    cisglmstate = cis_scan(X, y, G, bim, family, cis_list)

    # TODO: need write direct perm as child class
    adj_p, beta_res = perm(
        X,
        y,
        G,
        bim,
        jnp.min(cisglmstate.p),
        family,
        key_init,
        cis_list,
        sig_level,
        max_perm_direct,
        max_perm_beta,
    )

    return cisglmstate, adj_p, beta_res[0], beta_res[1]
