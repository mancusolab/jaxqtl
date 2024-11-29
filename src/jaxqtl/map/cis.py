from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import jax
import jax.random as rdm

from jax import numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from ..families.distribution import ExponentialFamily
from ..infer.glm import GLM
from ..infer.permutation import BetaPerm, InferBeta, InferBetaGLM
from ..infer.stderr import ErrVarEstimation, FisherInfoError, HuberError
from ..infer.utils import CisGLMState, HypothesisTest, ScoreTest
from ..io.readfile import ReadyDataState
from ..log import get_log
from ..post.qvalue import add_qvalues
from .utils import _get_geno_info, _setup_G_y


@dataclass
class MapCisSingleState:
    cisglm: CisGLMState
    pval_beta: Array
    beta_param: Array
    opt_status: Array
    true_nc: Array

    def get_lead(self, key: rdm.PRNGKey, random_tiebreak: bool = False) -> Tuple[List, int]:
        """Get lead SNP result for each gene

        :param key: randomly pick a SNP as lead SNP if there is tie when random_tiebreak=True
        :param random_tiebreak: `True` if randomly pick a lead SNP when there is tie, `False` if pick the first occurrence, default to `False`
        :return: lead SNP results and lead SNP index
        """
        # call lead eQTL
        if random_tiebreak:
            # randomly break tie
            key, split_key = rdm.split(key)
            ties_ind = jnp.argwhere(self.cisglm.p == jnp.nanmin(self.cisglm.p))  # return (k, 1)
            vdx = rdm.choice(split_key, ties_ind, (1,), replace=False)
        else:
            # take first occurrence
            vdx = int(jnp.nanargmin(self.cisglm.p))

        beta_1, beta_2, beta_converged = self.beta_param
        result = [
            beta_1,
            beta_2,
            beta_converged,
            jnp.array(self.opt_status),
            jnp.array(self.true_nc),
            self.cisglm.p[vdx],
            self.cisglm.beta[vdx],
            self.cisglm.se[vdx],
            self.pval_beta,
            self.cisglm.alpha[vdx],
            self.cisglm.converged[vdx],  # if wald test, this full model converged or not; if score, then cov-model
        ]

        result = [element.tolist() for element in result]

        return result, vdx


def map_cis(
    dat: ReadyDataState,
    family: ExponentialFamily,
    test: HypothesisTest = ScoreTest(),
    beta_estimator: InferBeta = InferBetaGLM(),
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
    compute_qvalue: bool = False,
    max_iter: int = 1000,
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
    :param window: window size (bp) of one side for cis scope, default to 500000, meaning in total 1Mb from left to right
    :param random_tiebreak: `True` if randomly pick a lead SNP when there is tie, `False` if pick the first occurrence, default to `False`
    :param sig_level: alpha significance level at each SNP level (not used), default to 0.05
    :param fdr_level: FDR level specified for across genes, default to 0.05 (not used if compute_qvalue=`False`)
    :param pi0: specified probability of null (optional) when compute_qvalue=`True`
    :param qvalue_lambda: an array of lambda value to fit a smooth spline (Optional)
    :param offset_eta: offset values when fitting regression for Negative Bionomial and Poisson, deault to 0s
    :param n_perm: number of permutation to estimate min p distribution for each gene using beta approximation approach, default to 1000
    :param robust_se: `True` if use huber white robust estimator for standard errors for nominal mapping (not used here), default to `False`
    :param compute_qvalue: `True` if add qvalue for genes, default to `False`
    :param max_iter: maximum iterations for fitting GLM, default to 500
    :param verbose: `True` if report QTL mapping progress in log file, default to `True`
    :param log: logger for QTL progress
    :return: data frame of QTL mapping results
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
        "opt_status",
        "true_nc",
        "pval_nominal",
        "slope",
        "slope_se",
        "pval_beta",
        "alpha_cov",
        "model_converged",
    ]

    results = []
    se_estimator = HuberError() if robust_se else FisherInfoError()

    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = max(0, start_min - window)
        rend = end_max + window

        # pull cis G and y for this gene
        G, y, var_df = _setup_G_y(dat, gene_name, str(chrom), lstart, rend)

        # skip if no cis SNPs found
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
            se_estimator,
            n_perm,
            test,
            beta_estimator,
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

        jax.clear_caches()  # Clear all compilation and staging caches
        result_out = _prepare_cis_result(
            G,
            chrom,
            gene_name,
            key,
            random_tiebreak,
            result,
            start_min,
            var_df,
            X,
            y,
            family,
            offset_eta,
            se_estimator,
            max_iter,
        )
        results.append(result_out)

    # filter results based on user specification (e.g., report all, report top, etc)
    result_df = pd.DataFrame.from_records(results, columns=out_columns)

    if compute_qvalue:
        result_df = add_qvalues(result_df, log, fdr_level, pi0, qvalue_lambda)

    return result_df


def _prepare_cis_result(
    G,
    chrom,
    gene_name,
    key,
    random_tiebreak,
    result,
    start_min,
    var_df,
    X,
    y,
    family,
    offset_eta,
    se_estimator,
    max_iter,
):
    """Get lead SNPs and their information

    :param G: genotype array
    :param chrom: chromosome number
    :param gene_name: gene name
    :param key: randomly pick a SNP as lead SNP if there is tie when random_tiebreak=`True`
    :param random_tiebreak: `True` if randomly pick a lead SNP when there is tie, `False` if pick the first occurrence, default to `False`
    :param result: data frame of QTL mapping result
    :param start_min: TSS start (0-based)
    :param var_df: data frame of variant information (bim)
    :return:
    """
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

    # fit full eQTL model for lead SNP
    glm = GLM(family=family, max_iter=max_iter)
    g = G[:, vdx]
    M = jnp.hstack((X, g[:, jnp.newaxis]))
    eta, alpha_n = glm.calc_eta_and_dispersion(M, y, offset_eta)
    glmstate = glm.fit(
        M,
        y,
        offset_eta=offset_eta,
        init=eta,
        alpha_init=alpha_n,
        se_estimator=se_estimator,
    )

    row[6] = glmstate.beta[-1].item()
    row[7] = glmstate.se[-1].item()

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
    se_estimator: ErrVarEstimation = FisherInfoError(),
    n_perm: int = 1000,
    test: HypothesisTest = ScoreTest(),
    beta_estimator: InferBeta = InferBetaGLM(),
    max_iter: int = 1000,
) -> MapCisSingleState:
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
    :param n_perm: number of permutation to estimate min p distribution for each gene using beta approximation approach, default to 1000
    :param test: approach for hypothesis test, default to ScoreTest()
    :param max_iter: maximum iterations for fitting GLM, default to 500
    :return: cis mapping results for a single gene
    """
    cisglmstate = test(X, G, y, family, offset_eta, se_estimator, max_iter)

    beta_key, direct_key = rdm.split(key_init)

    # if we -always- use BetaPerm now, we may as well drop the class aspect and
    # call function directly...
    # note: set max_perm_direct will change the parent class parameter
    perm = BetaPerm(max_perm_direct=n_perm)
    obs_p = jnp.nanmin(cisglmstate.p)
    obs_z = cisglmstate.z[int(jnp.nanargmin(cisglmstate.p))]

    pval_beta, beta_param, true_nc, opt_status = perm(
        X,
        y,
        G,
        obs_p,
        obs_z,
        family,
        beta_key,
        sig_level,
        offset_eta,
        test,
        se_estimator,
        beta_estimator,
        max_iter,
    )

    return MapCisSingleState(
        cisglm=cisglmstate, pval_beta=pval_beta, beta_param=beta_param, opt_status=opt_status, true_nc=true_nc
    )


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
