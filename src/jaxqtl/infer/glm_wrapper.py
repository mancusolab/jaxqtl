from jax import numpy as jnp

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.permutation import BetaPerm, Permutation
from jaxqtl.infer.utils import _setup_X_y, cis_GLM, cis_window_cutter
from jaxqtl.io.readfile import CleanDataState

# from .utils import cis_window_cutter


def run_cis_GLM(
    dat: CleanDataState,
    family: ExponentialFamily,
    gene_name: str,
    key_init,
    W: int = 1000000,
    sig_level=0.05,
    perm: Permutation = BetaPerm(),
    max_perm_direct=100,
    max_perm_beta=100,
):
    X, y = _setup_X_y(dat, gene_name)
    G = dat.genotype
    # TODO: prepare index list in cis window
    cis_list = cis_window_cutter(G, gene_name, dat.var_info, W)
    # cis_list = G.columns[0:100]
    glmres = cis_GLM(X, y, G, family, cis_list)

    _, adj_p, beta_k, beta_n = perm(
        X,
        y,
        G,
        jnp.min(glmres.p),
        family,
        key_init,
        cis_list,
        sig_level,
        max_perm_direct,
        max_perm_beta,
    )

    return glmres, adj_p, beta_k, beta_n
