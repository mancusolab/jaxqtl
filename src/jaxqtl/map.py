import os
from dataclasses import dataclass
from typing import List, NamedTuple, Tuple

import numpy as np
import pandas as pd

import jax.random as rdm
from jax import Array, numpy as jnp
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.permutation import BetaPerm, DirectPerm, Permutation
from jaxqtl.infer.utils import CisGLMState, _setup_G_y, cis_scan
from jaxqtl.io.readfile import ReadyDataState
from jaxqtl.log import get_log


@dataclass
class MapCisSingleState:
    cisglm: CisGLMState
    pval_perm: Array
    pval_beta: Array
    beta_param: Array

    def get_lead(self) -> Tuple[List, int]:

        # need to break tie
        # for now return first occurence

        vdx = int(jnp.argmin(self.cisglm.p))

        beta_1, beta_2, beta_converged = self.beta_param
        result = [
            beta_1,
            beta_2,
            beta_converged,
            self.cisglm.ma_samples[vdx],
            self.cisglm.ma_count[vdx],
            self.cisglm.af[vdx],
            self.cisglm.p[vdx],
            self.cisglm.beta[vdx],
            self.cisglm.se[vdx],
            self.pval_perm,
            self.pval_beta,
        ]

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


# write this assuming this is bulk data
def map_cis(
    dat: ReadyDataState,
    family: ExponentialFamily,
    append_intercept: bool = True,
    standardize: bool = True,
    seed: int = 123,
    window: int = 500000,
    sig_level: float = 0.05,
    perm: Permutation = BetaPerm(),
    verbose: bool = True,
) -> pd.DataFrame:

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
        "ma_samples",
        "ma_count",
        "af",
        "pval_nominal",
        "slope",
        "slope_se",
        "pval_perm",
        "pval_beta",
    ]

    results = []
    gene_info.gene_map = gene_info.gene_map.loc[gene_info.gene_map.chr == "22"]
    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = max(0, start_min - window)
        rend = end_max + window

        # pull cis G and y for this gene
        G, y, var_df = _setup_G_y(dat, gene_name, str(chrom), lstart, rend)

        # skip if no cis SNPs found
        if G.shape[1] == 0:
            continue

        key, g_key = rdm.split(key)
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
            perm,
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
        row, vdx = result.get_lead()

        # pull SNP info at lead hit index
        snp_id = var_df.iloc[vdx].snp
        snp_pos = var_df.iloc[vdx].pos
        tss_distance = snp_pos - start_min

        # combine lead hit info and gene meta data
        num_var_cis = G.shape[1]
        result = [gene_name, chrom, num_var_cis, snp_id, tss_distance] + row
        results.append(result)

        # unit test for 2 genes
        if len(results) > 1:
            break

    # filter results based on user speicification (e.g., report all, report top, etc)
    result_df = pd.DataFrame.from_records(results, columns=out_columns)

    return result_df


def map_cis_single(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    key_init: rdm.PRNGKey,
    sig_level: float = 0.05,
    perm: Permutation = BetaPerm(),
) -> MapCisSingleState:
    """Generate result of GLM for variants in cis
    For given gene, find all variants in + and - window size TSS region

    window: width of flanking on either side of TSS
    sig_level: desired significance level (not used)
    perm: Permutation method
    """

    cisglmstate = cis_scan(X, G, y, family)
    beta_key, direct_key = rdm.split(key_init)

    pval_beta, beta_param = perm(
        X,
        y,
        G,
        jnp.min(cisglmstate.p),
        family,
        beta_key,
        sig_level,
    )

    perm_iters_required = round(1 / sig_level)
    directperm = DirectPerm(perm_iters_required)
    pval_perm, _ = directperm(
        X,
        y,
        G,
        jnp.min(cisglmstate.p),
        family,
        direct_key,
        sig_level,
    )

    return MapCisSingleState(
        cisglm=cisglmstate,
        pval_perm=pval_perm,
        pval_beta=pval_beta,
        beta_param=beta_param,
    )


def map_cis_nominal(
    dat: ReadyDataState,
    family: ExponentialFamily,
    window: int = 500000,
) -> MapCisOutState:
    n, k = dat.covar.shape
    gene_info = dat.pheno_meta

    # append genotype as the last column
    X = jnp.hstack((jnp.ones((n, 1)), dat.covar))

    slope = []
    slope_se = []
    nominal_p = []
    converged = []
    num_var_cis = []
    gene_mapped_list = pd.DataFrame(columns=["gene_name", "chrom", "tss"])
    var_df_all = pd.DataFrame(
        columns=["chrom", "snp", "cm", "pos", "a0", "a1", "i", "phenotype_id", "tss"]
    )

    # TODO: fix for efficiency, filter by chromosome that exists in bim file
    gene_info.gene_map = gene_info.gene_map.loc[gene_info.gene_map.chr == "22"]

    for gene in gene_info:
        gene_name, chrom, start_min, end_max = gene
        lstart = max(0, start_min - window)
        rend = end_max + window

        # pull cis G and y for this gene
        G, y, var_df = _setup_G_y(dat, gene_name, str(chrom), lstart, rend)

        # skip if no cis SNPs found
        if G.shape[1] == 0:
            continue

        result = cis_scan(X, G, y, family)

        var_df["phenotype_id"] = gene_name
        var_df["tss"] = start_min
        var_df_all = pd.concat([var_df_all, var_df], ignore_index=True)
        gene_mapped_list.loc[len(gene_mapped_list)] = [gene_name, chrom, start_min]

        # combine results
        slope.append(result.beta)
        slope_se.append(result.se)
        nominal_p.append(result.p)
        converged.append(result.converged)
        num_var_cis.append(var_df.shape[0])

        # unit test for 3 genes
        if len(slope) > 1:
            break

    # filter results based on user speicification (e.g., report all, report top, etc)
    return MapCisOutState(
        slope=slope,
        slope_se=slope_se,
        nominal_p=nominal_p,
        pval_beta=[],
        beta_param=[],
        pval_perm=[],
        converged=converged,
        var_leading_df=var_df_all,
        gene_mapped_list=gene_mapped_list,
        num_var_cis=num_var_cis,
    )


def write_nominal(res: MapCisOutState, dat: ReadyDataState, out_dir: str, prefix):
    """write to parquet file by chrom for efficient data storage and retrieval"""

    start_row = 0
    end_row = 0
    outdf = res.var_leading_df
    outdf["tss_distance"] = outdf["pos"] - outdf["tss"]
    outdf = outdf.drop(["cm", "a0", "a1", "tss"], axis=1)

    outdf["af"] = np.NaN
    outdf["ma_samples"] = np.NaN
    outdf["ma_count"] = np.NaN
    outdf["pval_nominal"] = np.NaN
    outdf["slope"] = np.NaN
    outdf["slope_se"] = np.NaN

    # calculate genotype info
    G = dat.geno
    G = G[:, outdf.i.tolist()]

    n2 = 2 * G.shape[0]

    for idx in range(len(G.T)):
        g = G.T[idx]
        af = np.sum(g) / n2
        outdf["af"][idx] = af
        if af <= 0.5:
            outdf["ma_samples"][idx] = np.sum(
                g > 0.5
            )  # Number of samples carrying at least on minor allele
            outdf["ma_count"][idx] = np.sum(g[g > 0.5])  # Number of minor alleles
        else:
            outdf["ma_samples"][idx] = np.sum(g < 1.5)
            outdf["ma_count"][idx] = n2 - np.sum(g[g > 0.5])

    outdf = outdf.drop(["i"], axis=1)

    # TODO: add genotype info including af, ma_samples, ma_count
    for idx, _ in res.gene_mapped_list.iterrows():
        end_row += res.num_var_cis[idx]
        outdf["pval_nominal"][start_row:end_row] = res.nominal_p[idx]
        outdf["slope"][start_row:end_row] = res.slope[idx]
        outdf["slope_se"][start_row:end_row] = res.slope_se[idx]
        start_row = end_row

    # split by chrom
    for chrom in outdf["chrom"].unique().tolist():
        one_chrom_df = outdf.loc[outdf["chrom"] == chrom]
        one_chrom_df.to_parquet(
            os.path.join(out_dir, f"{prefix}.cis_qtl_pairs.{chrom}.parquet")
        )


def prepare_cis_output(dat: ReadyDataState, res: MapCisOutState):
    """Return nominal p-value, allele frequencies, etc. as pd.Series"""
    outdf = pd.DataFrame(
        columns=[
            "phenotype_id",
            "num_var",
            "beta_shape1",
            "beta_shape2",
            "true_df",
            "pval_true_df",
            "variant_id",
            "tss_distance",
            "ma_samples",
            "ma_count",
            "af",
            "pval_nominal",
            "slope",
            "slope_se",
            "pval_perm",
            "pval_beta",
        ]
    )

    n2 = 2 * dat.geno.shape[0]  # 2 * n_sample

    for idx, _ in res.gene_mapped_list.iterrows():
        g = np.array(dat.geno[:, res.var_leading_df.i[idx]])
        af = np.sum(g) / n2
        if af <= 0.5:
            ma_samples = np.sum(
                g > 0.5
            )  # Number of samples carrying at least on minor allele
            ma_count = np.sum(g[g > 0.5])  # Number of minor alleles
        else:
            ma_samples = np.sum(g < 1.5)
            ma_count = n2 - np.sum(g[g > 0.5])

        outdf.loc[idx] = [
            res.gene_mapped_list.gene_name[idx],
            res.num_var_cis[idx],
            res.beta_param[idx][0],
            res.beta_param[idx][1],
            np.NaN,
            np.NaN,
            res.var_leading_df.snp[idx],
            res.var_leading_df.pos[idx] - res.gene_mapped_list.tss[idx],
            ma_samples,
            ma_count,
            af,
            res.nominal_p[idx],
            res.slope[idx],
            res.slope_se[idx],
            res.pval_perm[idx],
            res.pval_beta[idx],
        ]

    return outdf
