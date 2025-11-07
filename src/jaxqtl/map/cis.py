from typing import List, Optional, Tuple

import pandas as pd

import equinox as eqx
import jax
import jax.random as rdm

from jax import numpy as jnp
from jaxtyping import ArrayLike, PRNGKeyArray

from ..infer.glm import GLM
from ..infer.permutations import AbstractPermutation, PermutationResult
from ..infer.utils import HypothesisTest, ScoreTest, TestResult
from ..io.readfile import ReadyDataState
from ..log import get_log
from ..post.qvalue import add_qvalues
from .utils import _get_geno_info, _setup_G_y


def map_cis(
    dat: ReadyDataState,
    glm: GLM,
    test: HypothesisTest,
    perm_test: AbstractPermutation,
    append_intercept: bool = True,
    standardize: bool = True,
    seed: int = 123,
    window: int = 500000,
    sig_level: float = 0.05,
    fdr_level: float = 0.05,
    pi0: Optional[float] = None,
    qvalue_lambda: Optional[ArrayLike] = None,
    offset: ArrayLike = 0.0,
    compute_qvalue: bool = False,
    verbose: bool = True,
    log=None,
) -> pd.DataFrame:
    """Cis eQTL mapping for each gene, report lead variant

    Run cis-eQTL mapping by fitting specified GLM model, such as Poisson and Negative Binomial.
    To test association between each SNP and gene expression, choose either score test (much faster) or
    wald test.
    For each gene, calculate the corrected p value using permutation to estimate the null distribution of
    minimum p values.

    :param dat: data input containing genotype array, bim, gene count data, gene meta data (tss), and covariates
    :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
    :param test: approach for hypothesis test, default to ScoreTest()
    :param append_intercept: `True` if want to append intercept, `False` otherwise
    :param standardize: `True` if want to standardize covariates data
    :param seed: seed for permutation, default to 123
    :param window: window size (bp) of one side for cis scope, default to 500000,
        meaning in total 1Mb from left to right
    :param random_tiebreak: `True` if randomly pick a lead SNP when there is tie, `False` if pick the first occurrence
        default to `False`
    :param sig_level: alpha significance level at each SNP level (not used), default to 0.05
    :param fdr_level: FDR level specified for across genes, default to 0.05 (not used if compute_qvalue=`False`)
    :param pi0: specified probability of null (optional) when compute_qvalue=`True`
    :param qvalue_lambda: an array of lambda value to fit a smooth spline (Optional)
    :param offset_eta: offset values when fitting regression for Negative Bionomial and Poisson, deault to 0s
    :param n_perm: number of permutation to estimate min p distribution for each gene using beta approximation approach
        default to 1000
    :param compute_qvalue: `True` if add qvalue for genes, default to `False`
    :param verbose: `True` if report QTL mapping progress in log file, default to `True`
    :param log: logger for QTL progress
    :return: data frame of QTL mapping results
    """
    if log is None:
        log = get_log()

    # TODO: we need to do some validation here...
    X = dat.covar
    n, k = X.shape

    if standardize:
        X = X / jnp.std(X, axis=0)

    if append_intercept:
        X = jnp.hstack((jnp.ones((n, 1)), X))

    key = rdm.key(seed)

    results = CisResults()
    for i, gene in enumerate(dat):
        gene_name, chrom, start_min, end_max = gene
        lstart = max(0, start_min - window)
        rend = end_max + window

        # pull cis G and y for this gene
        G, y, var_df = _setup_G_y(dat, gene_name, str(chrom), lstart, rend)

        # skip if no cis SNPs found
        if G.shape[1] == 0:
            if verbose:
                log.warning(f"No cis-SNPs found for {gene_name} over region {chrom}:{lstart}-{rend}. Skipping.")
            continue

        if verbose:
            log.info(f"Performing cis-qtl scan for {gene_name} over region {chrom}:{lstart}-{rend}")

        key, g_key = rdm.split(key, 2)
        test_result, perm_result = map_cis_single(X, G, y, offset, perm_test, test, g_key, sig_level)

        if verbose:
            log.info(f"Finished cis-qtl scan for {gene_name} over region {chrom}:{lstart}-{rend}")

        # clear caches every 50 genes
        if (i + 1) % 50 == 0:
            jax.clear_caches()  # clear up caches

        results.add_result(
            X,
            G,
            y,
            offset,
            glm,
            test,
            test_result,
            perm_result,
            chrom,
            gene_name,
            start_min,
            var_df,
            key,
        )

    result_df = results.as_df()

    if compute_qvalue:
        result_df = add_qvalues(result_df, log, fdr_level, pi0, qvalue_lambda)

    return result_df


@eqx.filter_jit
def map_cis_single(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    offset_eta: ArrayLike,
    perm: AbstractPermutation,
    test: HypothesisTest,
    key: rdm.PRNGKey,
    sig_level: float = 0.05,
) -> tuple[TestResult, PermutationResult]:
    """Fit GLM for SNP-gene pairs and report results

    :rtype: MapCisSingleState
    :param X: array of covariates
    :param G: genotype array
    :param y: gene expression array
    :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
    :param key_init: key for jax RNG
    :param sig_level: alpha significance level at each SNP level (not used), default to 0.05
    :param offset_eta: offset values when fitting regression for Negative Bionomial and Poisson, deault to 0s
    :param se_estimator: SE estimator using HuberError() or FisherInfoError()
    :param n_perm: number of permutation to estimate min p distribution for each gene using beta approximation approach
        default to 1000
    :param test: approach for hypothesis test, default to ScoreTest()
    :return: cis mapping results for a single gene
    """
    test_result = test(X, G, y, offset_eta)

    perm_result = perm(
        X,
        G,
        y,
        offset_eta,
        test_result,
        test,
        key,
        sig_level,
    )
    return test_result, perm_result


def write_parqet(outdf: pd.DataFrame, method: str, out_path: str):
    """write parquet file for nominal scan (split by chr)

    :param outdf: data frame of full cis nominal mapping
    :param method: wald or score
    :param out_path: output path
    :return: None
    """
    # split by chrom
    for chrom in outdf["chrom"].unique().tolist():
        one_chrom_df = outdf.loc[outdf["chrom"] == chrom]
        one_chrom_df.drop("i", axis=1, inplace=True)  # remove index i
        one_chrom_df.to_parquet(out_path + f".cis_qtl_pairs.{chrom}.{method}.parquet")

    return


class CisResults:
    out_columns = [
        "phenotype_id",
        "chrom",
        "num_var",
        "variant_id",
        "a1",
        "a0",
        "pos",
        "tss_distance",
        "ma_count",
        "af",
        "beta_shape1",
        "beta_shape2",
        "beta_converged",
        "opt_status",
        "nc_estimate",
        "effect",
        "effect_se",
        "pval_nominal",
        "pval_adj",
        "adj_method",
        "alpha",
        "model_converged",
    ]

    def __init__(self):
        self.results = {cname: [] for cname in self.out_columns}

    def _get_lead(self, result: TestResult, perm_result: PermutationResult, key: PRNGKeyArray) -> Tuple[List, int]:
        """Get lead SNP result for each gene

        :param key: randomly pick a SNP as lead SNP if there is tie when random_tiebreak=True
        :param random_tiebreak: `True` if randomly pick a lead SNP when there is tie,
            `False` if pick the first occurrence, default to `False`
        :return: lead SNP results and lead SNP index
        """
        # randomly break tie
        minp = jnp.nanmin(result.p)
        ties_ind = jnp.argwhere(result.p == minp).squeeze()  # why does this add extra axis?
        if ties_ind.ndim > 0:
            vdx = rdm.choice(key, ties_ind, replace=False)
        else:
            vdx = ties_ind

        adj_pvalue, aux = perm_result

        # this is kind of hacky but if aux is not None we did a beta-approximation
        if aux is not None:
            beta_params, nc_estimate, opt_status = aux
            beta_k = float(beta_params.k)
            beta_n = float(beta_params.n)
            beta_converged = beta_params.converged
            opt_status = bool(opt_status)
            nc_estimate = float(nc_estimate)
            method = "BETA"
        else:
            beta_k = float("nan")
            beta_n = float("nan")
            beta_converged = True
            opt_status = True
            nc_estimate = float("nan")
            method = "ACAT"

        result = {
            "beta_shape1": beta_k,
            "beta_shape2": beta_n,
            "beta_converged": beta_converged,
            "opt_status": opt_status,
            "nc_estimate": nc_estimate,
            "effect": float(result.beta[vdx]),
            "effect_se": float(result.se[vdx]),
            "pval_nominal": float(result.p[vdx]),
            "pval_adj": float(adj_pvalue[vdx]),
            "adj_method": method,
            "alpha": float(result.alpha[vdx]),
            "model_converged": bool(result.converged[vdx]),
        }
        # if wald test, this full model converged or not; if score, then cov-model
        return result, vdx

    def add_result(
        self,
        X,
        G,
        y,
        offset,
        glm,
        test,
        test_result,
        perm_result,
        chrom,
        gene_name,
        start_min,
        variant_df,
        key,
    ):
        """Get lead SNPs and their information

        :param G: genotype array
        :param chrom: chromosome number
        :param gene_name: gene name
        :param key: randomly pick a SNP as lead SNP if there is tie when random_tiebreak=`True`
        :param random_tiebreak: `True` if randomly pick a lead SNP when there is tie,
            `False` if pick the first occurrence default to `False`
        :param result: data frame of QTL mapping result
        :param start_min: TSS start (0-based)
        :param variant_df: data frame of variant information (bim)
        :return:
        """
        g_info = _get_geno_info(G)

        # get info at lead hit, and lead hit index
        row, vdx = self._get_lead(test_result, perm_result, key)

        # pull SNP info at lead hit index
        vdx = int(vdx)
        af = g_info.af[vdx]
        ma_count = g_info.ma_count[vdx]
        snp_id = variant_df.iloc[vdx].snp
        snp_pos = variant_df.iloc[vdx].pos
        a1 = variant_df.iloc[vdx].a1
        a0 = variant_df.iloc[vdx].a0
        tss_distance = snp_pos - start_min

        # combine lead hit info and gene meta data
        num_var_cis = G.shape[1]

        # fit full eQTL model for lead SNP
        if isinstance(test, ScoreTest):
            g = G[:, vdx]
            M = jnp.hstack((X, g[:, jnp.newaxis]))
            glmstate = glm.fit(M, y, offset)
            row["effect"] = float(glmstate.beta[-1])
            row["effect_se"] = float(glmstate.se[-1])

        meta = [gene_name, chrom, num_var_cis, snp_id, a1, a0, snp_pos, tss_distance, ma_count, af]
        # this assumes that the order of results in row are correct wrt the col names
        for idx, cname in enumerate(self.out_columns[: len(meta)]):
            self.results[cname].append(meta[idx])

        for cname, value in row.items():
            self.results[cname].append(value)

        return

    def as_df(self):
        return pd.DataFrame(self.results)
