from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

import jax.random as rdm
from jax import Array, numpy as jnp
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily

# from jaxqtl.infer.glm import GLM
from jaxqtl.infer.permutation import BetaPerm, BetaPermScore
from jaxqtl.infer.utils import (
    CisGLMScoreState,
    CisGLMState,
    _setup_G_y,
    cis_scan,
    cis_scan_intercept_only,
    cis_scan_score,
)
from jaxqtl.io.readfile import ReadyDataState
from jaxqtl.log import get_log
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
                self.cisglm.p == self.cisglm.p.min()
            )  # return (k, 1)
            vdx = rdm.choice(split_key, ties_ind, (1,), replace=False)
        else:
            # take first occurrence
            vdx = int(jnp.argmin(self.cisglm.p))

        beta_1, beta_2, beta_converged = self.beta_param
        result = [
            beta_1,
            beta_2,
            beta_converged,
            self.cisglm.ma_count[vdx],
            self.cisglm.af[vdx],
            self.cisglm.p[vdx],
            self.cisglm.beta[vdx],
            self.cisglm.se[vdx],
            self.pval_beta,
        ]

        result = [element.tolist() for element in result]

        return result, vdx


@dataclass
class MapCisSingleScoreState:
    cisglm: CisGLMScoreState
    pval_beta: Array
    beta_param: Array

    def get_lead(
        self, key: rdm.PRNGKey, random_tiebreak: bool = False
    ) -> Tuple[List, int]:
        # break tie to call lead eQTL
        if random_tiebreak:
            key, split_key = rdm.split(key)
            ties_ind = jnp.argwhere(
                self.cisglm.p == self.cisglm.p.min()
            )  # return (k, 1)
            vdx = rdm.choice(split_key, ties_ind, (1,), replace=False)
        else:
            # take first occurrence
            vdx = int(jnp.argmin(self.cisglm.p))

        beta_1, beta_2, beta_converged = self.beta_param
        result = [
            beta_1,
            beta_2,
            beta_converged,
            self.cisglm.p[vdx],
            self.cisglm.Z[vdx],
            self.pval_beta,
            self.cisglm.alpha[vdx],
        ]

        result = [element.tolist() for element in result]

        return result, vdx


class MapCisOutState(NamedTuple):
    slope: List
    slope_se: List
    nominal_p: List
    pval_beta: List
    beta_param: List
    pval_perm: List
    converged: List
    num_var_cis: List
    var_leading_df: pd.DataFrame
    gene_mapped_list: pd.DataFrame


class MapCisScoreOutState(NamedTuple):
    Z: List
    nominal_p: List
    pval_beta: List
    beta_param: List
    pval_perm: List
    converged: List
    num_var_cis: List
    var_leading_df: pd.DataFrame
    gene_mapped_list: pd.DataFrame


class _GenoInfo(NamedTuple):
    af: Array
    ma_count: Array


# write this assuming this is bulk data
def map_cis(
    dat: ReadyDataState,
    family: ExponentialFamily,
    append_intercept: bool = True,
    standardize: bool = True,
    seed: int = 123,
    window: int = 500000,
    random_tiebreak: bool = False,
    sig_level: float = 0.05,
    verbose: bool = True,
    fdr_level: float = 0.05,
    pi0: Optional[float] = None,
    qvalue_lambda: np.ndarray = None,
    offset_eta: ArrayLike = 0.0,
    robust_se: bool = True,
    n_perm: int = 1000,
    compute_qvalue: bool = True,
    log=None,
) -> pd.DataFrame:
    """Cis mapping for each gene, report lead variant
    use permutation to determine cis-eQTL significance level (direct permutation + beta distribution method)
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
        "beta_shape1",
        "beta_shape2",
        "beta_converged",
        "ma_count",
        "af",
        "pval_nominal",
        "slope",
        "slope_se",
        "pval_beta",
    ]

    results = []

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
            X, G, y, family, g_key, sig_level, offset_eta, robust_se, n_perm
        )

        if verbose:
            log.info(
                "Finished cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )
        # get info at lead hit, and lead hit index
        row, vdx = result.get_lead(key, random_tiebreak)

        # pull SNP info at lead hit index
        snp_id = var_df.iloc[vdx].snp
        snp_pos = var_df.iloc[vdx].pos
        tss_distance = snp_pos - start_min

        # combine lead hit info and gene meta data
        num_var_cis = G.shape[1]
        result_out = [gene_name, chrom, num_var_cis, snp_id, tss_distance] + row
        results.append(result_out)

    # filter results based on user speicification (e.g., report all, report top, etc)
    result_df = pd.DataFrame.from_records(results, columns=out_columns)

    if compute_qvalue:
        result_df = add_qvalues(result_df, log, fdr_level, pi0, qvalue_lambda)

    return result_df


def map_cis_score(
    dat: ReadyDataState,
    family: ExponentialFamily,
    append_intercept: bool = True,
    standardize: bool = True,
    seed: int = 123,
    window: int = 500000,
    random_tiebreak: bool = False,
    sig_level: float = 0.05,
    verbose: bool = True,
    fdr_level: float = 0.05,
    pi0: Optional[float] = None,
    qvalue_lambda: Optional[np.ndarray] = None,
    offset_eta: ArrayLike = 0.0,
    n_perm: int = 1000,
    compute_qvalue: bool = True,
    log=None,
) -> pd.DataFrame:
    """Cis mapping for each gene, report lead variant
    use permutation to determine cis-eQTL significance level (direct permutation + beta distribution method)
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
        "Z",
        "pval_beta",
        "alpha_cov",
    ]

    results = []

    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = max(0, start_min - window)
        rend = end_max + window

        # pull cis G and y for this gene
        G, y, var_df = _setup_G_y(dat, gene_name, str(chrom), lstart, rend)

        # skip if no cis SNPs found
        if (
            G.shape[1] == 0
        ):  # TODO: double check that you have a non-None G that has 2-dim when no cis-SNPs exist
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

        result = map_cis_single_score(
            X, G, y, family, g_key, sig_level, offset_eta, n_perm, log
        )
        if verbose:
            log.info(
                "Finished cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

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
        results.append(result_out)

    # filter results based on user specification (e.g., report all, report top, etc)
    result_df = pd.DataFrame.from_records(results, columns=out_columns)

    if compute_qvalue:
        result_df = add_qvalues(result_df, log, fdr_level, pi0, qvalue_lambda)

    return result_df


def map_cis_single(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    key_init: rdm.PRNGKey,
    sig_level: float = 0.05,
    offset_eta: ArrayLike = 0.0,
    robust_se: bool = True,
    n_perm: int = 1000,
) -> MapCisSingleState:
    """Generate result of GLM for variants in cis
    For given gene, find all variants in + and - window size TSS region

    window: width of flanking on either side of TSS
    sig_level: desired significance level (not used)
    perm: Permutation method
    """

    cisglmstate = cis_scan(X, G, y, family, offset_eta, robust_se)

    beta_key, direct_key = rdm.split(key_init)

    # if we -alwaays- use BetaPerm now, we may as well drop the class aspect and
    # call function directly...
    perm = BetaPerm(max_perm_direct=n_perm)
    pval_beta, beta_param = perm(
        X,
        y,
        G,
        jnp.min(cisglmstate.p),
        family,
        beta_key,
        sig_level,
        offset_eta,
        robust_se,
    )

    return MapCisSingleState(
        cisglm=cisglmstate,
        pval_beta=pval_beta,
        beta_param=beta_param,
    )


def map_cis_single_score(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    key_init: rdm.PRNGKey,
    sig_level: float = 0.05,
    offset_eta: ArrayLike = 0.0,
    n_perm: int = 1000,
    log=None,
) -> MapCisSingleScoreState:
    """Generate result of GLM for variants in cis
    For given gene, find all variants in + and - window size TSS region

    window: width of flanking on either side of TSS
    sig_level: desired significance level (not used)
    perm: Permutation method
    """
    cisglmstate = cis_scan_score(X, G, y, family, offset_eta)

    beta_key, direct_key = rdm.split(key_init)

    # if we -always- use BetaPerm now, we may as well drop the class aspect and
    # call function directly...
    perm = BetaPermScore(max_perm_direct=n_perm)
    pval_beta, beta_param = perm(
        X, y, G, jnp.min(cisglmstate.p), family, beta_key, sig_level, offset_eta, log
    )

    return MapCisSingleScoreState(
        cisglm=cisglmstate,
        pval_beta=pval_beta,
        beta_param=beta_param,
    )


def map_cis_nominal(
    dat: ReadyDataState,
    family: ExponentialFamily,
    log=None,
    append_intercept: bool = True,
    standardize: bool = True,
    window: int = 500000,
    verbose: bool = True,
    offset_eta: ArrayLike = 0.0,
    robust_se: bool = True,
):
    """eQTL Mapping for all cis-SNP gene pairs

    append_intercept: add a column of ones in front for intercepts in design matrix
    standardize: on covariates

    Returns:
        write out parquet file by chrom for efficient data storage and retrieval
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

    af = []
    ma_count = []
    slope = []
    slope_se = []
    nominal_p = []
    converged = []
    num_var_cis = []
    gene_mapped_list = pd.DataFrame(columns=["gene_name", "chrom", "tss"])
    var_df_all = pd.DataFrame(
        columns=["chrom", "snp", "cm", "pos", "a0", "a1", "i", "phenotype_id", "tss"]
    )

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

        if verbose:
            log.info(
                "Performing cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        result = cis_scan(X, G, y, family, offset_eta, robust_se)
        g_info = _get_geno_info(G)
        if verbose:
            log.info(
                "Finished cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        var_df["phenotype_id"] = gene_name
        var_df["tss"] = start_min
        var_df_all = pd.concat([var_df_all, var_df], ignore_index=True)
        gene_mapped_list.loc[len(gene_mapped_list)] = [gene_name, chrom, start_min]

        # combine results
        af.append(g_info.af)
        ma_count.append(g_info.ma_count)

        slope.append(result.beta)
        slope_se.append(result.se)
        nominal_p.append(result.p)
        converged.append(result.converged)
        num_var_cis.append(var_df.shape[0])

    # write result
    start_row = 0
    end_row = 0
    outdf = var_df_all
    outdf["tss_distance"] = outdf["pos"] - outdf["tss"]
    outdf = outdf.drop(["cm"], axis=1)

    # add additional columns
    outdf["af"] = np.NaN
    outdf["ma_count"] = np.NaN
    outdf["pval_nominal"] = np.NaN
    outdf["slope"] = np.NaN
    outdf["slope_se"] = np.NaN
    outdf["converged"] = np.NaN

    for idx, _ in gene_mapped_list.iterrows():
        end_row += num_var_cis[idx]
        outdf.loc[np.arange(start_row, end_row), "af"] = af[idx]
        outdf.loc[np.arange(start_row, end_row), "ma_count"] = ma_count[idx]
        outdf.loc[np.arange(start_row, end_row), "pval_nominal"] = nominal_p[idx]
        outdf.loc[np.arange(start_row, end_row), "slope"] = slope[idx]
        outdf.loc[np.arange(start_row, end_row), "slope_se"] = slope_se[idx]
        outdf.loc[np.arange(start_row, end_row), "converged"] = converged[idx]
        start_row = end_row

    return outdf


def map_cis_nominal_score(
    dat: ReadyDataState,
    family: ExponentialFamily,
    log=None,
    append_intercept: bool = True,
    standardize: bool = True,
    window: int = 500000,
    verbose: bool = True,
    offset_eta: ArrayLike = 0.0,
):
    """eQTL Mapping for all cis-SNP gene pairs

    append_intercept: add a column of ones in front for intercepts in design matrix
    standardize: on covariates

    Returns:
        score test statistics and p value (no effect estimates)
        write out parquet file by chrom for efficient data storage and retrieval
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

    af = []
    ma_count = []
    Z = []
    nominal_p = []
    converged = []
    num_var_cis = []
    alpha_cov = []
    gene_mapped_list = pd.DataFrame(columns=["gene_name", "chrom", "tss"])
    var_df_all = pd.DataFrame(
        columns=["chrom", "snp", "cm", "pos", "a0", "a1", "i", "phenotype_id", "tss"]
    )

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

        if verbose:
            log.info(
                "Performing cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        result = cis_scan_score(X, G, y, family, offset_eta)
        g_info = _get_geno_info(G)

        if verbose:
            log.info(
                "Finished cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        var_df["phenotype_id"] = gene_name
        var_df["tss"] = start_min
        var_df_all = pd.concat([var_df_all, var_df], ignore_index=True)
        gene_mapped_list.loc[len(gene_mapped_list)] = [gene_name, chrom, start_min]

        # combine results
        af.append(g_info.af)
        ma_count.append(g_info.ma_count)

        nominal_p.append(result.p)
        Z.append(result.Z)
        converged.append(result.converged)
        num_var_cis.append(var_df.shape[0])
        alpha_cov.append(result.alpha)

    # write result
    start_row = 0
    end_row = 0
    outdf = var_df_all
    outdf["tss_distance"] = outdf["pos"] - outdf["tss"]
    outdf = outdf.drop(["cm"], axis=1)

    # add additional columns
    outdf["af"] = np.NaN
    outdf["ma_count"] = np.NaN
    outdf["pval_nominal"] = np.NaN
    outdf["Z"] = np.NaN
    outdf["converged"] = np.NaN
    outdf["alpha_cov"] = np.NaN

    for idx, _ in gene_mapped_list.iterrows():
        end_row += num_var_cis[idx]
        outdf.loc[np.arange(start_row, end_row), "af"] = af[idx]
        outdf.loc[np.arange(start_row, end_row), "ma_count"] = ma_count[idx]
        outdf.loc[np.arange(start_row, end_row), "pval_nominal"] = nominal_p[idx].T
        outdf.loc[np.arange(start_row, end_row), "Z"] = Z[idx].T
        outdf.loc[np.arange(start_row, end_row), "converged"] = converged[idx]
        outdf.loc[np.arange(start_row, end_row), "alpha_cov"] = alpha_cov[idx]
        start_row = end_row

    return outdf


def write_parqet(outdf: pd.DataFrame, method: str, out_path: str):
    """
    write parquet file for nominal scan (split by chr)
    """
    # split by chrom
    for chrom in outdf["chrom"].unique().tolist():
        one_chrom_df = outdf.loc[outdf["chrom"] == chrom]
        one_chrom_df.drop("i", axis=1, inplace=True)  # remove index i
        one_chrom_df.to_parquet(out_path + f".cis_qtl_pairs.{chrom}.{method}.parquet")


def map_fit_intercept_only(
    dat: ReadyDataState,
    family: ExponentialFamily,
    log=None,
    verbose: bool = True,
    offset_eta: ArrayLike = 0.0,
):
    """fit intercept only model for each gene and output fitted values

    Returns:
        write out tsv file by chrom for efficient data storage and retrieval
    """
    if log is None:
        log = get_log()

    # TODO: we need to do some validation here...
    n, k = dat.covar.shape
    X = jnp.ones((n, 1))  # intercept only

    if verbose:
        log.info("Begin mapping")

    result = cis_scan_intercept_only(X, jnp.array(dat.pheno.count), family, offset_eta)

    if verbose:
        log.info("Finished mapping")

    # write result
    outdf = pd.DataFrame.from_records(result.T)
    outdf.columns = dat.pheno.count.columns

    return outdf


def _get_geno_info(G: ArrayLike) -> _GenoInfo:
    n, p = G.shape
    counts = jnp.sum(G, axis=0)  # count REF allele
    af = counts / (2.0 * n)
    flag = af <= 0.5
    ma_counts = jnp.where(flag, counts, 2 * n - counts)

    return _GenoInfo(af, ma_counts)
