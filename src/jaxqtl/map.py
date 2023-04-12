import os
from typing import List, NamedTuple

import numpy as np
import pandas as pd

import jax.random as rdm
from jax import Array, numpy as jnp
from jax.typing import ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.permutation import BetaPerm, DirectPerm, Permutation
from jaxqtl.infer.utils import CisGLMState, _setup_G_y, cis_scan
from jaxqtl.io.readfile import ReadyDataState


class MapCis_SingleState(NamedTuple):
    cisglm: CisGLMState
    pval_perm: Array
    pval_beta: Array
    beta_param: Array


class MapCis_OutState(NamedTuple):
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
    seed: int = 123,
    window: int = 500000,
    sig_level: float = 0.05,
    perm: Permutation = BetaPerm(),
) -> MapCis_OutState:
    n, k = dat.covar.shape
    gene_info = dat.pheno_meta

    # append genotype as the last column
    X = jnp.hstack((jnp.ones((n, 1)), dat.covar))
    key = rdm.PRNGKey(seed)

    slope = []
    slope_se = []
    nominal_p = []
    pval_beta = []
    pval_perm = []
    beta_param = []
    converged = []
    num_var_cis = []
    gene_mapped_list = pd.DataFrame(columns=["gene_name", "chrom", "tss"])
    var_leading_df = pd.DataFrame(
        columns=["chrom", "snp", "cm", "pos", "a0", "a1", "i"]
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

        key, g_key = rdm.split(key)

        result = map_cis_single(
            X,
            G,
            y,
            family,
            g_key,
            sig_level,
            perm,
        )

        # need to break tie
        # for now return first occurence
        nominal_p_onegene = result.cisglm.p
        leading_var_idx = nominal_p_onegene.tolist().index(jnp.min(nominal_p_onegene))

        var_leading_df.loc[len(var_leading_df)] = var_df.iloc[leading_var_idx]

        gene_mapped_list.loc[len(gene_mapped_list)] = [gene_name, chrom, start_min]

        # combine results
        slope.append(result.cisglm.beta[leading_var_idx])
        slope_se.append(result.cisglm.se[leading_var_idx])
        nominal_p.append(result.cisglm.p[leading_var_idx])
        pval_beta.append(result.pval_beta)
        pval_perm.append(result.pval_perm)
        beta_param.append(result.beta_param)
        converged.append(result.cisglm.converged[leading_var_idx])
        num_var_cis.append(var_df.shape[0])

        # unit test for 2 genes
        if len(pval_beta) > 1:
            break

    # filter results based on user speicification (e.g., report all, report top, etc)

    return MapCis_OutState(
        slope=slope,
        slope_se=slope_se,
        nominal_p=nominal_p,
        pval_beta=pval_beta,
        beta_param=beta_param,
        pval_perm=pval_perm,
        converged=converged,
        var_leading_df=var_leading_df,
        gene_mapped_list=gene_mapped_list,
        num_var_cis=num_var_cis,
    )


def map_cis_single(
    X: ArrayLike,
    G: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    key_init,
    sig_level=0.05,
    perm: Permutation = BetaPerm(),
) -> MapCis_SingleState:
    """Generate result of GLM for variants in cis
    For given gene, find all variants in + and - window size TSS region

    window: width of flanking on either side of TSS
    sig_level: desired significance level (not used)
    perm: Permutation method
    """

    cisglmstate = cis_scan(X, G, y, family)

    pval_beta, beta_param = perm(
        X,
        y,
        G,
        jnp.min(cisglmstate.p),
        family,
        key_init,
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
        key_init,
        sig_level,
    )

    return MapCis_SingleState(
        cisglm=cisglmstate,
        pval_perm=pval_perm,
        pval_beta=pval_beta,
        beta_param=beta_param,
    )


def map_cis_nominal(
    dat: ReadyDataState,
    family: ExponentialFamily,
    window: int = 500000,
) -> MapCis_OutState:
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
    return MapCis_OutState(
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


def write_nominal(res: MapCis_OutState, out_dir: str, prefix):
    """write to parquet file by chrom for efficient data storage and retrieval"""

    start_row = 0
    end_row = 0
    outdf = res.var_leading_df
    outdf["tss_distance"] = outdf["pos"] - outdf["tss"]
    outdf = outdf.drop(["cm", "i", "a0", "a1", "tss"], axis=1)

    outdf["af"] = np.NaN
    outdf["ma_samples"] = np.NaN
    outdf["ma_count"] = np.NaN
    outdf["pval_nominal"] = np.NaN
    outdf["slope"] = np.NaN
    outdf["slope_se"] = np.NaN

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


def prepare_cis_output(dat: ReadyDataState, res: MapCis_OutState):
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
