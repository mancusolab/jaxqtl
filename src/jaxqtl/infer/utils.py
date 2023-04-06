from typing import List, NamedTuple, Tuple

import jax.lax as lax
from jax import numpy as jnp
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.glm import GLM
from jaxqtl.io.readfile import CleanDataState


class CisGLMState(NamedTuple):
    beta: jnp.ndarray
    se: jnp.ndarray
    p: jnp.ndarray
    num_iters: jnp.ndarray
    converged: jnp.ndarray
    var: List


def _cis_window_cutter(dat: CleanDataState, chrom: str, start: int, end: int) -> List:
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
    # gene_info = "./example/data/ensembl_allgenes.txt"
    # gene_map = pd.read_csv(gene_info, delimiter="\t")
    # gene_map.columns = [
    #     "chr",
    #     "gene_start",
    #     "gene_end",
    #     "symbol",
    #     "tss_start",
    #     "strand",
    #     "gene_type",
    #     "ensemble_id",
    #     "refseq_id",
    # ]
    # gene_map["tss_left_end"] = gene_map["tss_start"] - window  # it's ok to be negative
    # gene_map["tss_right_end"] = gene_map["tss_start"] + window
    #
    #
    # # TODO: need check if gene exist in both strand
    # query = gene_map[gene_map.ensemble_id == gene_name]
    # starts_min = query["tss_left_end"].min()
    # starts_max = query["tss_right_end"].max()

    var_info = dat.bim
    cis_var_info = var_info.loc[
        (var_info["chrom"] == chrom)
        & (var_info["pos"] >= start)
        & (var_info["pos"] <= end)
    ]

    # return column indices
    return cis_var_info.i


def _setup_X_y(dat: CleanDataState, gene_name: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    prepare allocation for X and y
    example: gene_name = 'ENSG00000030582'
    """
    G = dat.genotype  # n x p variants data frame
    covar = jnp.asarray(dat.covar)
    nobs, _ = G.shape
    num_params = covar.shape[1] + 2  # covariate features + one SNP + intercept

    # Xmat: intercept, SNP, cov1, cov2, ...
    Xmat = jnp.ones((nobs, num_params))
    Xmat = Xmat.at[:, 2:].set(covar)

    # get count vector for this gene
    ycount = dat.count.X[:, dat.count.var.index == gene_name].astype("float64")

    return Xmat, ycount


def _setup_G_y(dat: CleanDataState, gene_name: str, chrom: str, start: int, end: int):
    G = _cis_window_cutter(dat, chrom, start, end)
    ycount = dat.count[gene_name]

    return G, ycount


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
