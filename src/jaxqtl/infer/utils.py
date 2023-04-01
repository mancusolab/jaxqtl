# import genomicranges
from typing import Tuple

import pandas as pd

from jax import numpy as jnp
from jax._src.basearray import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.glm import GLM, GLMState
from jaxqtl.io.readfile import CleanDataState


# TODO: find strand of imputed variant
# TODO: use which gene database, eg. emsembl? (onek1k has this identifier)
def cis_window_cutter(W, gene_start, gene_end, var_list):
    """
    return variant list in cis for given gene
    """
    gene_info = "./example/data/list_genes_qc.all.txt"
    df = pd.read_csv(gene_info, delimiter="\t")
    df = df[
        [
            "chr",
            "ensembl.start",
            "ensembl.end",
            "ensembl.strand",
            "name",
            "ensembl.ENSG",
        ]
    ]
    # format it as: seqnames, starts, end, strand
    df.columns = ["seqnames", "starts", "ends", "old_strand", "symbol", "ensembl_id"]
    gr_strand = ["+", "+", "-", "-", "*", "*", "*"]
    df_strand = ["1", "1|1", "-1", "-1|-1", "-1|1", "1|-1", "1|1|-1"]

    df["strand"] = df["old_strand"].replace(df_strand, gr_strand)

    # gr = genomicranges.fromPandas(df)  # convert to gr object

    return


def _setup_X_y(dat: CleanDataState, gene_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    G = dat.genotype  # n x p variants
    covar = dat.covar
    nobs, _ = G.shape
    num_params = covar.shape[1] + 2  # covariate features + one SNP + intercept

    # Xmat: intercept, SNP, cov1, cov2, ...
    Xmat = jnp.ones((nobs, num_params))
    Xmat = Xmat.at[:, 2:].set(covar)

    ycount = dat.count.X[:, gene_idx].astype("float64")

    return Xmat, ycount


def _update_X(X: ArrayLike, G: ArrayLike, var_idx: int) -> jnp.ndarray:
    """ "
    create design matrix for a variant, i.e. replace variant column
    """
    return X.at[:, 1].set(G[:, var_idx])


def cis_GLM(
    X: ArrayLike, y: ArrayLike, G: ArrayLike, family: ExponentialFamily, cis_window
):
    """
    run GLM across variants in a flanking window of given gene
    cis-widow: plus and minus W base pairs, total length 2*cis_window
    """

    all_beta = jnp.zeros((cis_window,))
    all_se = jnp.zeros((cis_window,))
    all_pval = jnp.zeros((cis_window,))
    all_num_iters = jnp.zeros((cis_window,))
    all_converged = jnp.zeros((cis_window,))

    for idx in range(cis_window):
        X = X.at[:, 1].set(G[:, idx])
        glmstate = GLM(
            X=X,
            y=y,
            family=family,
            append=False,
            maxiter=1000,
        ).fit()

        all_beta = all_beta.at[idx].set(glmstate.beta[1])
        all_se = all_se.at[idx].set(glmstate.se[1])
        all_pval = all_pval.at[idx].set(glmstate.p[1])
        all_num_iters = all_num_iters.at[idx].set(glmstate.num_iters)
        all_converged = all_converged.at[idx].set(glmstate.converged)

    return GLMState(all_beta, all_se, all_pval, all_num_iters, all_converged)
