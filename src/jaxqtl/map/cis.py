from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import jax.random as rdm
from jax import numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.permutation import BetaPerm
from jaxqtl.infer.utils import CisGLMState, HypothesisTest, ScoreTest
from jaxqtl.io.readfile import ReadyDataState
from jaxqtl.log import get_log
from jaxqtl.map.utils import _get_geno_info, _setup_G_y
from jaxqtl.post.qvalue import add_qvalues


@dataclass
class MapCisSingleState:
    cisglm: CisGLMState
    pval_beta: Array
    beta_param: Array

    def get_lead(
        self, key: rdm.PRNGKey, random_tiebreak: bool = False
    ) -> Tuple[List, int]:
        # break tie to call lead eQTL
        if random_tiebreak:
            key, split_key = rdm.split(key)
            ties_ind = jnp.argwhere(
                self.cisglm.p == jnp.nanmin(self.cisglm.p)
            )  # return (k, 1)
            vdx = rdm.choice(split_key, ties_ind, (1,), replace=False)
        else:
            # take first occurrence
            vdx = int(jnp.nanargmin(self.cisglm.p))

        beta_1, beta_2, beta_converged = self.beta_param
        result = [
            beta_1,
            beta_2,
            beta_converged,
            self.cisglm.p[vdx],
            self.cisglm.beta[vdx],
            self.cisglm.se[vdx],
            self.pval_beta,
            self.cisglm.alpha[vdx],
            self.cisglm.converged[
                vdx
            ],  # if wald test, this full model converged or not; if score, then cov-model
        ]

        result = [element.tolist() for element in result]

        return result, vdx


def map_cis(
    dat: ReadyDataState,
    family: ExponentialFamily,
    test: HypothesisTest = ScoreTest(),
    append_intercept: bool = True,
    standardize: bool = True,
    seed: int = 123,
    window: int = 500000,
    random_tiebreak: bool = False,
    sig_level: float = 0.05,
    fdr_level: float = 0.05,
    pi0: Optional[float] = None,
    qvalue_lambda: Optional[np.ndarray] = None,
    offset_eta: ArrayLike = 0.0,
    n_perm: int = 1000,
    robust_se: bool = False,
    compute_qvalue: bool = True,
    max_iter: int = 500,
    verbose: bool = True,
    log=None,
) -> pd.DataFrame:
    """Cis mapping for each gene, report lead variant
    use permutation to determine cis-eQTL significance level (direct permutation + beta distribution method)
    score test: fit null once and compute TS
    """
    if log is None:
        log = get_log()

    # TODO: we need to do some validation here...
    X = dat.covar
    n, k = X.shape

    gene_info = dat.pheno_meta

    # append genotype as the last column
    if standardize:
        X = X / jnp.std(X, axis=0)

    if append_intercept:
        X = jnp.hstack((jnp.ones((n, 1)), X))

    key = rdm.PRNGKey(seed)

    out_columns = [
        "phenotype_id",
        "chrom",
        "num_var",
        "variant_id",
        "tss_distance",
        "ma_count",
        "af",
        "beta_shape1",
        "beta_shape2",
        "beta_converged",
        "pval_nominal",
        "slope",
        "slope_se",
        "pval_beta",
        "alpha_cov",
        "model_converged",
    ]

    results = []

    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = max(0, start_min - window)
        rend = end_max + window

        # pull cis G and y for this gene
        G, y, var_df = _setup_G_y(dat, gene_name, str(chrom), lstart, rend)

        # skip if no cis SNPs found
        # TODO: double check that you have a non-None G that has 2-dim when no cis-SNPs exist
        if G.shape[1] == 0:
            if verbose:
                log.info(
                    "No cis-SNPs found for %s over region %s:%s-%s. Skipping.",
                    gene_name,
                    str(chrom),
                    str(lstart),
                    str(rend),
                )
            continue

        key, g_key = rdm.split(key, 2)
        if verbose:
            log.info(
                "Performing cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        result = map_cis_single(
            X,
            G,
            y,
            family,
            g_key,
            sig_level,
            offset_eta,
            robust_se,
            n_perm,
            test,
            max_iter,
        )
        if verbose:
            log.info(
                "Finished cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        result_out = _prepare_cis_result(
            G, chrom, gene_name, key, random_tiebreak, result, start_min, var_df
        )
        results.append(result_out)

    # filter results based on user specification (e.g., report all, report top, etc)
    result_df = pd.DataFrame.from_records(results, columns=out_columns)

    if compute_qvalue:
        result_df = add_qvalues(result_df, log, fdr_level, pi0, qvalue_lambda)

    return result_df


def _prepare_cis_result(
    G, chrom, gene_name, key, random_tiebreak, result, start_min, var_df
):
    g_info = _get_geno_info(G)
    # get info at lead hit, and lead hit index
    row, vdx = result.get_lead(key, random_tiebreak)
    # pull SNP info at lead hit index
    af = g_info.af[vdx]
    ma_count = g_info.ma_count[vdx]
    snp_id = var_df.iloc[vdx].snp
    snp_pos = var_df.iloc[vdx].pos
    tss_distance = snp_pos - start_min
    # combine lead hit info and gene meta data
    num_var_cis = G.shape[1]
    result_out = [
        gene_name,
        chrom,
        num_var_cis,
        snp_id,
        tss_distance,
        ma_count,
        af,
    ] + row
    return result_out


def map_cis_single(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    key_init: rdm.PRNGKey,
    sig_level: float = 0.05,
    offset_eta: ArrayLike = 0.0,
    robust_se: bool = False,
    n_perm: int = 1000,
    test: HypothesisTest = ScoreTest(),
    max_iter: int = 500,
) -> MapCisSingleState:
    """Generate result of GLM for variants in cis
    For given gene, find all variants in + and - window size TSS region

    window: width of flanking on either side of TSS
    sig_level: desired significance level (not used)
    perm: Permutation method
    """
    cisglmstate = test(X, G, y, family, offset_eta, robust_se, max_iter)

    beta_key, direct_key = rdm.split(key_init)

    # if we -always- use BetaPerm now, we may as well drop the class aspect and
    # call function directly...
    perm = BetaPerm(max_perm_direct=n_perm)
    pval_beta, beta_param = perm(
        X,
        y,
        G,
        jnp.nanmin(cisglmstate.p),
        family,
        beta_key,
        sig_level,
        offset_eta,
        test,
        robust_se,
        max_iter,
    )

    return MapCisSingleState(
        cisglm=cisglmstate,
        pval_beta=pval_beta,
        beta_param=beta_param,
    )


def write_parqet(outdf: pd.DataFrame, method: str, out_path: str):
    """
    write parquet file for nominal scan (split by chr)
    """
    # split by chrom
    for chrom in outdf["chrom"].unique().tolist():
        one_chrom_df = outdf.loc[outdf["chrom"] == chrom]
        one_chrom_df.drop("i", axis=1, inplace=True)  # remove index i
        one_chrom_df.to_parquet(out_path + f".cis_qtl_pairs.{chrom}.{method}.parquet")

    return
