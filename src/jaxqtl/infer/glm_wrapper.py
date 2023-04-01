from jax import numpy as jnp

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.permutation import BetaPerm, Permutation
from jaxqtl.infer.utils import _setup_X_y, cis_GLM
from jaxqtl.io.readfile import CleanDataState

# from .utils import cis_window_cutter


def run_cis_GLM(
    dat: CleanDataState,
    family: ExponentialFamily,
    key_init,
    gene_idx=0,
    cis_idx=10,
    sig_level=0.05,
    perm: Permutation = BetaPerm(),
    max_perm_direct=100,
    max_perm_beta=100,
):
    X, y = _setup_X_y(dat, gene_idx)
    G = dat.genotype
    glmres = cis_GLM(X, y, G, family, cis_idx)

    _, adj_p, beta_k, beta_n = perm(
        X,
        y,
        G,
        jnp.min(glmres.p),
        family,
        key_init,
        gene_idx,
        cis_idx,
        sig_level,
        max_perm_direct,
        max_perm_beta,
    )

    return glmres, adj_p, beta_k, beta_n
