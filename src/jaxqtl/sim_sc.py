import argparse as ap
import logging
import os
import sys

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import qtl.norm

from pandas_plink import read_plink
from tensorqtl import cis

import jax
import jax.numpy as jnp
import jax.random as rdm

from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from jaxqtl.families.distribution import (
    ExponentialFamily,
    Gaussian,
    NegativeBinomial,
    Poisson,
)
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.permutation import InferBetaGLM
from jaxqtl.infer.stderr import FisherInfoError
from jaxqtl.infer.utils import CommonTest, score_test_snp, ScoreTest
from jaxqtl.log import get_logger
from jaxqtl.map.cis import map_cis_single
from jaxqtl.map.utils import _ACAT


class SimState(NamedTuple):
    X: Array  # design matrix contains intercept + covariates + genotype
    y: Array
    beta: Array  # true betas
    sim_scdat: Array
    h2obs: Array


class SimResState(NamedTuple):
    pval_nb_wald: Array
    pval_pois_wald: Array
    pval_lm_score: Array
    pval_lm_wald: Array
    pval_tqtl: Array
    pval_nb_score: Array
    pval_pois_score: Array
    sc_mean_ct: Array
    sc_express_percent: Array
    sc_libsize_valid: Array
    bulk_mean_ct: Array
    bulk_express_percent: Array
    bulk_libsize_valid: Array
    h2obs: Array
    alpha: Array
    pval_nb_pbeta: Array
    pval_nb_pacat: Array


def sim_data(
    libsize: pd.DataFrame,
    nobs: int,
    sample_covar: ArrayLike,
    g: Optional[ArrayLike],  # shape nx1
    family: ExponentialFamily = Poisson(),
    maf: float = 0.3,
    beta0: float = 1.0,  # intercept determine baseline counts
    seed: int = 1,
    V_a: float = 0.1,
    V_re: float = 0.2,
    V_disp: float = 0.0,
    m_causal: int = 1,
    baseline_mu: float = 0.0,
) -> SimState:
    p = 2  # if covar not specified, then only intercept + genotype
    X = jnp.ones((nobs, 1))  # intercept

    key = rdm.PRNGKey(seed)

    # append sample level covariates (age, sex) but will zero effects to run saigeqtl
    num_covar = 2  # age, sex
    p += num_covar

    # bootstrap for nobs number of individuals
    sample_covar_nobs = sample_covar.sample(n=nobs, replace=True, random_state=seed)
    sample_covar_nobs.loc[:, 'iid'] = np.arange(1, sample_covar_nobs.shape[0] + 1)

    X_cov = jnp.array(sample_covar_nobs[['age', 'sex']])
    X_cov = (X_cov - X_cov.mean(axis=0)) / X_cov.std(axis=0)
    X = jnp.column_stack((X, X_cov))

    beta_shape = (p, 1)
    beta = jnp.ones(beta_shape)
    beta = beta.at[0].set(beta0)

    # set effect of age and sex as zero
    beta_covar = jnp.zeros((num_covar, 1))  # fix covariate effect to be 0
    beta = beta.at[1 : p - 1].set(beta_covar)

    # geno in shape of nx1
    if g is None:
        key, snp_key = rdm.split(key, 2)
        g = rdm.binomial(snp_key, 2, maf, shape=(nobs, 1))  # genotype (0,1,2)

    X = jnp.column_stack((X, g))  # include genotype at last column

    key, g_key = rdm.split(key, 2)
    g_beta = rdm.normal(g_key) * np.sqrt(V_a / m_causal) if V_a > 0 else 0.0
    beta = beta.at[-1].set(g_beta)  # put genotype as last column

    # sample random effect of each individual
    key, re_key = rdm.split(key, 2)
    bi = rdm.normal(re_key, (nobs, 1)) * np.sqrt(V_re) if V_re > 0 else 0

    # merge for individual cell library size in sampled individuals
    libsize = libsize.merge(sample_covar_nobs[['individual', 'iid']], 'right', on="individual")

    sample_covar_nobs.drop(['age', 'sex'], axis=1, inplace=True)
    covar_std = pd.DataFrame({'iid': np.arange(1, sample_covar_nobs.shape[0] + 1), 'age': X[:, 1], 'sex': X[:, 2]})
    sample_covar_nobs = covar_std.merge(sample_covar_nobs, 'right', on="iid")

    # standardize age and sex
    libsize.drop(['age', 'sex'], axis=1, inplace=True)
    libsize = covar_std.merge(libsize, 'right', on="iid")

    # broad cast individual level eta to cell level
    eta_df = pd.DataFrame({'eta': (X @ beta + bi).ravel(), 'iid': np.arange(1, sample_covar_nobs.shape[0] + 1)})
    eta_df.loc[:, 'iid'] = sample_covar_nobs['iid'].values
    eta_df = eta_df.merge(libsize, 'right', on="iid")

    # compute eta_i + bi + log(offset_ij)
    # eta_df['log_offset'] = 0 # set offset to zero
    eta = jnp.array(eta_df[['eta', 'log_offset']]).sum(axis=1)
    mu = family.glink.inverse(eta)

    # for each individual mu_i, broadcast to num_cells (Poisson model)
    key, y_key = rdm.split(key, 2)
    y = rdm.poisson(y_key, mu)  # long vector of individual cell
    y = y.reshape(-1, 1)

    h2obs = _calc_h2obs(V_a, V_disp, V_re, baseline_mu)

    return SimState(jnp.array(X), jnp.array(y), jnp.array(beta), eta_df, h2obs)


def _calc_h2obs(V_a: float, V_disp: float, V_re: float, baseline_mu: float) -> Array:
    # Calculate heritability of additive genetics on observed scale
    tot_var = V_a + V_re + V_disp
    lamb = np.exp(baseline_mu + tot_var / 2.0)
    h2g_obs = lamb * V_a / (lamb * (np.exp(tot_var) - 1) + 1)
    return jnp.array(h2g_obs)


def simulate_genotypes(
    key: jax.Array, ld_matrix: ArrayLike, n_individuals: int, maf_range: tuple = (0.05, 0.5)
) -> Array:
    """
    Simulate genotype matrix (0/1/2) from a given LD matrix and fixed MAF.

    Parameters:
        key           : jax.random.PRNGKey
        ld_matrix     : (n_snps, n_snps) LD correlation matrix (symmetric, PSD)
        n_individuals : number of individuals to simulate
        maf           : fixed minor allele frequency for all SNPs (e.g., 0.2)

    Returns:
        genotypes     : (n_individuals, n_snps) matrix with values in {0, 1, 2}
    """

    n_snps = ld_matrix.shape[0]

    """Ensure LD matrix is PSD by adding a small ridge term."""
    ld_matrix = 0.5 * (ld_matrix + ld_matrix.T)  # ensure symmetry
    ld_matrix = ld_matrix + 1e-5 * jnp.eye(ld_matrix.shape[0])

    # Cholesky decomposition of LD matrix
    L = jnp.linalg.cholesky(ld_matrix)

    # Split PRNGKey for genotype simulation and MAFs
    key_geno, key_maf = rdm.split(key)

    # Generate latent standard normal variables
    Z = rdm.normal(key_geno, shape=(n_individuals, n_snps))
    latent = Z @ L.T  # inject LD structure

    # Generate random MAFs within range
    min_maf, max_maf = maf_range
    mafs = rdm.uniform(key_maf, shape=(n_snps,), minval=min_maf, maxval=max_maf)

    # Compute quantile thresholds under HWE
    t1 = norm.ppf((1 - mafs) ** 2)
    t2 = norm.ppf(1 - mafs**2)

    # Discretize latent Gaussian values into genotypes
    def discretize_column(g_col, t1j, t2j):
        return jnp.where(g_col < t1j, 0, jnp.where(g_col < t2j, 1, 2))

    genotypes = jax.vmap(discretize_column, in_axes=(1, 0, 0), out_axes=1)(latent, t1, t2)

    return genotypes


def run_sim(
    onek1k_libsize: pd.DataFrame,
    bim,
    bed,
    seed: int = 1,
    nobs: int = 1000,
    family: ExponentialFamily = Poisson(),
    maf: float = 0.3,
    beta0: float = 1.0,  # intercept determine baseline counts
    V_a: float = 0.1,
    V_re: float = 0.2,
    V_disp: float = 0.0,
    m_causal: int = 1,
    baseline_mu: float = 0.0,
    G: Optional[ArrayLike] = None,  # shape of num_sim x n
    LD_path: Optional[str] = None,
    sample_covar: Optional[ArrayLike] = None,  # nxp
    num_sim: int = 1000,
    write_sc: bool = True,
    out_path: Optional[str] = None,  # write out single cell data in saigeqtl format
    max_iter: int = 1000,
) -> SimResState:
    pval_nb_wald = jnp.array([])
    pval_nb_score = jnp.array([])

    pval_nb_pbeta = jnp.array([])
    pval_nb_pacat = jnp.array([])

    pval_pois_wald = jnp.array([])
    pval_pois_score = jnp.array([])

    pval_lm_wald = jnp.array([])
    pval_lm_score = jnp.array([])
    pval_tqtl = jnp.array([])

    sc_mean_ct_list = jnp.array([])
    sc_express_percent_list = jnp.array([])
    sc_libsize_valid_list = jnp.array([])
    bulk_mean_ct_list = jnp.array([])
    bulk_express_percent_list = jnp.array([])
    bulk_libsize_valid_list = jnp.array([])
    alpha_list = jnp.array([])

    if LD_path is not None:
        # run jaxQTL beta permutation method
        ld_mat = pd.read_csv(LD_path, header=None, sep="\t")
        ld_mat = jnp.array(ld_mat)

        family = NegativeBinomial()

        cisG_key = rdm.PRNGKey(seed)
        cisG = simulate_genotypes(key=cisG_key, ld_matrix=ld_mat, n_individuals=nobs, maf_range=(0.05, 0.5))
        cisG_key, map_key = rdm.split(cisG_key, 2)

    for i in range(num_sim):
        snp = None if G is None else G[i].reshape(-1, 1)

        # check constant y
        all_constant = True
        loop_idx = 0
        while all_constant:
            X, y, beta, sim_scdat, h2obs = sim_data(
                libsize=onek1k_libsize,
                nobs=nobs,
                g=snp,
                family=family,
                maf=maf,
                beta0=beta0,  # intercept determine baseline counts
                seed=i + seed + loop_idx,  # use simulation index for generating phenotype and sampling
                V_a=V_a,
                V_re=V_re,
                V_disp=V_disp,
                m_causal=m_causal,
                baseline_mu=baseline_mu,
                sample_covar=sample_covar,  # nxp
            )

            # create pheno for saigeqtl
            df_out = sim_scdat[['iid', 'log_offset', 'age', 'sex']]
            df_out.loc[:, 'libsize'] = jnp.exp(jnp.array(df_out['log_offset']))
            df_out.loc[:, 'gene'] = y.ravel()

            # convert to pseudo-bulk for jaxqtl and linear model
            # create summary statistics for simulated count
            df_bulk = df_out.groupby(['iid']).agg({'gene': 'sum', 'libsize': 'sum'})

            y_bulk = (jnp.array(df_bulk['gene'])).reshape(-1, 1)

            all_constant = (y == y[0]).all() | (y_bulk == y_bulk[0]).all()
            loop_idx += 1

        # write tsv in saigeqtl input format (for one gene as one replicate)
        if write_sc:
            df_out = df_out[['iid', 'log_offset', 'age', 'sex', 'gene']]
            df_out.to_csv(f"{out_path}.pheno{i+1}.tsv", sep="\t", index=False)

        sc_mean_ct = y.ravel().mean()
        sc_express_percent = (y.ravel() > 0).mean()
        sc_libsize = jnp.exp(jnp.array(df_out['log_offset'])).reshape(-1, 1)
        sc_libsize_valid = ((y / sc_libsize) <= 1.0).mean()

        bulk_mean_ct = y_bulk.ravel().mean()
        bulk_express_percent = (y_bulk.ravel() > 0).mean()
        bulk_libsize = jnp.array(df_bulk['libsize']).reshape(-1, 1)
        bulk_libsize_valid = (y_bulk / bulk_libsize <= 1.0).mean()

        sc_mean_ct_list = jnp.append(sc_mean_ct_list, sc_mean_ct)
        sc_express_percent_list = jnp.append(sc_express_percent_list, sc_express_percent)
        sc_libsize_valid_list = jnp.append(sc_libsize_valid_list, sc_libsize_valid)
        bulk_mean_ct_list = jnp.append(bulk_mean_ct_list, bulk_mean_ct)
        bulk_express_percent_list = jnp.append(bulk_express_percent_list, bulk_express_percent)
        bulk_libsize_valid_list = jnp.append(bulk_libsize_valid_list, bulk_libsize_valid)

        log_offset = jnp.log(bulk_libsize)
        jaxqtl_pois = GLM(family=Poisson())
        jaxqtl_nb = GLM(family=NegativeBinomial())
        jaxqtl_lm = GLM(family=Gaussian())

        # convert bulk int to float
        y_bulk = y_bulk.astype(jnp.float64)

        # fit poisson wald test
        init_pois = jaxqtl_pois.family.init_eta(y_bulk)
        glm_state_pois = jaxqtl_pois.fit(
            X, y_bulk, init=init_pois, offset_eta=log_offset, se_estimator=FisherInfoError()
        )

        pval_pois_wald = jnp.append(pval_pois_wald, glm_state_pois.p[-1])

        # fit NB wald test
        init_nb, alpha_n = jaxqtl_nb.calc_eta_and_dispersion(X, y_bulk, log_offset)
        alpha_n = jnp.nan_to_num(alpha_n, nan=0.1)

        glm_state_nb = jaxqtl_nb.fit(
            X, y_bulk, init=init_nb, alpha_init=alpha_n, offset_eta=log_offset, se_estimator=FisherInfoError()
        )

        pval_nb_wald = jnp.append(pval_nb_wald, glm_state_nb.p[-1])

        # fit lm (genexN); only one gene so don't need convert to cpm per individual
        norm_df = qtl.norm.inverse_normal_transform(pd.DataFrame(y_bulk).T)
        y_norm = np.array(norm_df.T)

        init_lm = jaxqtl_lm.family.init_eta(y_norm)
        glm_state = jaxqtl_lm.fit(X, y_norm, init=init_lm, se_estimator=FisherInfoError())

        pval_lm_wald = jnp.append(pval_lm_wald, glm_state.p[-1])

        # score test for poisson and NB
        X_cov = X[:, 0:-1]
        g = X[:, -1].reshape(-1, 1)

        glm_null_pois = jaxqtl_pois.fit(X_cov, y_bulk, init=init_pois, offset_eta=log_offset)
        _, pval, _, _ = score_test_snp(G=g, X=X_cov, glm_null_res=glm_null_pois)

        pval_pois_score = jnp.append(pval_pois_score, pval)

        init_nb, alpha_n = jaxqtl_nb.calc_eta_and_dispersion(X_cov, y_bulk, log_offset)
        alpha_n = jnp.nan_to_num(alpha_n, nan=0.1)

        glm_state_nb = jaxqtl_nb.fit(X_cov, y_bulk, init=init_nb, alpha_init=alpha_n, offset_eta=log_offset)
        _, pval, _, _ = score_test_snp(G=g, X=X_cov, glm_null_res=glm_state_nb)
        alpha_list = jnp.append(alpha_list, glm_state_nb.alpha)

        pval_nb_score = jnp.append(pval_nb_score, pval)

        glm_state_lm = jaxqtl_lm.fit(X_cov, y_bulk, init=init_lm)
        _, pval, _, _ = score_test_snp(G=g, X=X_cov, glm_null_res=glm_state_lm)

        pval_lm_score = jnp.append(pval_lm_score, pval)

        # run tensorqtl; G is pxn when reading in from read_plink
        # format for tensorqtl input
        genotype_df = pd.DataFrame(G, index=bim['snp'].values)  # pxn
        genotype_df.columns = [str(i) for i in range(1, bed.shape[1] + 1)]
        genotype_df.index.name = "snp"
        variant_df = bim[['snp', 'chrom', 'pos']]
        variant_df.set_index('snp', inplace=True)

        phenotype_df = (pd.DataFrame(y_norm)).T
        phenotype_df.columns = [str(i) for i in range(1, bed.shape[1] + 1)]
        genotype_df.index.name = "snp"
        phenotype_df.set_index(pd.Series(['gene']), inplace=True)
        phenotype_df.index.name = "gene_id"

        phenotype_pos_df = pd.DataFrame({'chr': str(1), 'pos': [500]})
        phenotype_pos_df.set_index(pd.Series(['gene']), inplace=True)
        phenotype_pos_df.index.name = "gene_id"
        prefix = f"{out_path}.pheno{i+1}"

        covariates_df = pd.DataFrame(X[:, 1:-1])
        covariates_df.columns = ['age', 'sex']
        covariates_df.set_index(pd.Series([str(i) for i in range(1, bed.shape[1] + 1)]), inplace=True)

        cis.map_nominal(
            genotype_df,
            variant_df,
            phenotype_df,
            phenotype_pos_df,
            prefix,
            covariates_df,
            output_dir='.',
            window=1000000,
        )

        tqtl_res = pd.read_parquet(f"{prefix}.cis_qtl_pairs.1.parquet")
        pval_tqtl = jnp.append(pval_tqtl, tqtl_res['pval_nominal'][i])
        os.remove(f"{prefix}.cis_qtl_pairs.1.parquet")

        # calulate gene-level p value, but only for simulation under null (Va=0)
        if LD_path is not None:
            # run jaxQTL beta permutation method
            pbeta_result = map_cis_single(
                X_cov,
                cisG,  # shape nxp
                y_bulk,
                family=family,
                key_init=map_key,
                offset_eta=log_offset,
                se_estimator=FisherInfoError(),
                n_perm=1000,
                test=ScoreTest(),
                beta_estimator=InferBetaGLM(),
                max_iter=max_iter,
            )
            pval_nb_pbeta = jnp.append(pval_nb_pbeta, pbeta_result.pval_beta)

            # run jaxQTL acat method
            test = ScoreTest()
            acat_result = test(X_cov, cisG, y_bulk, family, log_offset, FisherInfoError(), max_iter, CommonTest())
            pvec = acat_result.p
            acat_p = jnp.where(jnp.all(jnp.isnan(pvec)), jnp.nan, _ACAT(pvec[~jnp.isnan(pvec)]))  # check NaN
            pval_nb_pacat = jnp.append(pval_nb_pacat, acat_p)

            jax.clear_caches()

    return SimResState(
        pval_nb_wald=pval_nb_wald,
        pval_nb_score=pval_nb_score,
        pval_pois_wald=pval_pois_wald,
        pval_pois_score=pval_pois_score,
        pval_lm_wald=pval_lm_wald,
        pval_lm_score=pval_lm_score,
        pval_tqtl=pval_tqtl,
        sc_mean_ct=sc_mean_ct_list,
        sc_express_percent=sc_express_percent_list,
        sc_libsize_valid=sc_libsize_valid_list,
        bulk_mean_ct=bulk_mean_ct_list,
        bulk_express_percent=bulk_express_percent_list,
        bulk_libsize_valid=bulk_libsize_valid_list,
        h2obs=jnp.repeat(h2obs, num_sim),
        alpha=alpha_list,
        pval_nb_pbeta=pval_nb_pbeta,
        pval_nb_pacat=pval_nb_pacat,
    )


def main(args):
    argp = ap.ArgumentParser(description="")  # create an instance
    argp.add_argument("--geno", type=str, help="Genotype plink prefix, eg. chr17")
    argp.add_argument("--CT", type=str, help="cell type")
    argp.add_argument("--libsize-path", type=str, help="path to read in library size, with header")
    argp.add_argument("--nobs", type=int, help="Sample size")
    argp.add_argument("--m-causal", type=int, default=1, help="Number of causal variants")
    argp.add_argument("--model", type=str, choices=["gaussian", "poisson", "NB"], help="Model")
    argp.add_argument("--beta0", type=float, default=0, help="Intercept")
    argp.add_argument("--Va", type=float, default=0.1, help="Variance explained by genetics, 0 means eqtl_beta=0")
    argp.add_argument("--Vre", type=float, default=0.2, help="Variance explained by random effect (across individuals)")
    argp.add_argument("--Vdisp", type=float, default=0.0, help="Variance dispersion")
    argp.add_argument("--baseline-mu", type=float, default=0.0, help="population baseline mu on observed scale")
    argp.add_argument("--maf", type=float, default=0.1, help="MAF")
    argp.add_argument("--seed", type=int, default=1, help="seed")
    argp.add_argument("--num-sim", type=int, default=1000, help="Number of simulation, equal to #markers in plink file")
    argp.add_argument("--fwer", type=float, default=0.05)
    argp.add_argument("--LD-path", type=str, help="path to LD matrix file, no header tsv file")
    argp.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose for logger",
    )
    argp.add_argument(
        "--write-sc",
        action="store_true",
        default=False,
        help="Verbose for logger",
    )
    argp.add_argument("--out-sc", type=str, help="out file prefix for saigeqtl phenotype")
    argp.add_argument("--out", type=str, help="out file prefix")

    args = argp.parse_args(args)  # a list a strings

    platform = "cpu"
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", platform)

    log = get_logger(__name__, args.out)
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    if args.model == "poisson":
        family = Poisson()
    elif args.model == "NB":
        family = NegativeBinomial()
    elif args.model == "gaussian":
        family = Gaussian()
    else:
        log.info("Please choose either poisson or gaussian.")

    if args.geno is not None:
        # read in genotype file
        bim, fam, bed = read_plink(args.geno, verbose=False)
        G = bed.compute()  # array
        log.info("Read in genotype file.")
    else:
        G = None

    # read in observed library size
    onek1k = pd.read_csv(args.libsize_path, sep="\t")
    sample_covar = onek1k.drop_duplicates(subset=['individual', 'age', 'sex'], keep='last').reset_index(drop=True)
    log.info("sample library size from onek1k.")

    res = run_sim(
        onek1k_libsize=onek1k,
        bim=bim,
        bed=bed,
        seed=args.seed,
        nobs=args.nobs,
        family=family,
        maf=args.maf,
        beta0=args.beta0,  # intercept determine baseline counts
        V_a=args.Va,
        V_re=args.Vre,
        V_disp=args.Vdisp,
        m_causal=args.m_causal,
        baseline_mu=args.baseline_mu,
        G=G,
        LD_path=args.LD_path,
        sample_covar=sample_covar,
        num_sim=args.num_sim,
        write_sc=args.write_sc,
        out_path=args.out_sc,  # write out single cell data in saigeqtl format
    )

    d_rej = {
        'CT': args.CT,
        'maf': args.maf,
        'beta0': args.beta0,
        'Va': args.Va,
        'Vre': args.Vre,
        'nobs': args.nobs,
        'rep': args.num_sim,
        'rej_nb_wald': [jnp.mean(res.pval_nb_wald[~jnp.isnan(res.pval_nb_wald)] < args.fwer)],
        'rej_nb_score': [jnp.mean(res.pval_nb_score[~jnp.isnan(res.pval_nb_score)] < args.fwer)],
        'rej_pois_wald': [jnp.mean(res.pval_pois_wald[~jnp.isnan(res.pval_pois_wald)] < args.fwer)],
        'rej_pois_score': [jnp.mean(res.pval_pois_score[~jnp.isnan(res.pval_pois_score)] < args.fwer)],
        'rej_lm_wald': [jnp.mean(res.pval_lm_wald[~jnp.isnan(res.pval_lm_wald)] < args.fwer)],
        'rej_lm_score': [jnp.mean(res.pval_lm_score[~jnp.isnan(res.pval_lm_score)] < args.fwer)],
        'rej_tqtl': [jnp.mean(res.pval_tqtl[~jnp.isnan(res.pval_tqtl)] < args.fwer)],
        'sc_mean_ct': [(res.sc_mean_ct).mean()],
        'sc_express_percent': [(res.sc_express_percent).mean()],
        'sc_libsize_valid': [(res.sc_libsize_valid).mean()],
        'bulk_mean_ct': [(res.bulk_mean_ct).mean()],
        'bulk_express_percent': [(res.bulk_express_percent).mean()],
        'bulk_libsize_valid': [(res.bulk_libsize_valid).mean()],
        'alpha': [(res.alpha[~jnp.isnan(res.alpha)].mean())],
        'rej_nb_score_pbeta': [jnp.mean(res.pval_nb_pbeta[~jnp.isnan(res.pval_nb_pbeta)] < args.fwer)],
        'rej_nb_score_pacat': [jnp.mean(res.pval_nb_pacat[~jnp.isnan(res.pval_nb_pacat)] < args.fwer)],
    }

    if args.LD_path is not None:
        d_rej = {
            'CT': args.CT,
            'maf': args.maf,
            'beta0': args.beta0,
            'Va': args.Va,
            'Vre': args.Vre,
            'nobs': args.nobs,
            'rep': args.num_sim,
            'rej_nb_wald': [jnp.mean(res.pval_nb_wald[~jnp.isnan(res.pval_nb_wald)] < args.fwer)],
            'rej_nb_score': [jnp.mean(res.pval_nb_score[~jnp.isnan(res.pval_nb_score)] < args.fwer)],
            'rej_pois_wald': [jnp.mean(res.pval_pois_wald[~jnp.isnan(res.pval_pois_wald)] < args.fwer)],
            'rej_pois_score': [jnp.mean(res.pval_pois_score[~jnp.isnan(res.pval_pois_score)] < args.fwer)],
            'rej_lm_wald': [jnp.mean(res.pval_lm_wald[~jnp.isnan(res.pval_lm_wald)] < args.fwer)],
            'rej_lm_score': [jnp.mean(res.pval_lm_score[~jnp.isnan(res.pval_lm_score)] < args.fwer)],
            'rej_tqtl': [jnp.mean(res.pval_tqtl[~jnp.isnan(res.pval_tqtl)] < args.fwer)],
            'sc_mean_ct': [(res.sc_mean_ct).mean()],
            'sc_express_percent': [(res.sc_express_percent).mean()],
            'sc_libsize_valid': [(res.sc_libsize_valid).mean()],
            'bulk_mean_ct': [(res.bulk_mean_ct).mean()],
            'bulk_express_percent': [(res.bulk_express_percent).mean()],
            'bulk_libsize_valid': [(res.bulk_libsize_valid).mean()],
            'alpha': [(res.alpha[~jnp.isnan(res.alpha)].mean())],
            'rej_nb_score_pbeta': [jnp.mean(res.pval_nb_pbeta[~jnp.isnan(res.pval_nb_pbeta)] < args.fwer)],
            'rej_nb_score_pacat': [jnp.mean(res.pval_nb_pacat[~jnp.isnan(res.pval_nb_pacat)] < args.fwer)],
        }

        # write p value vector
        d_pval = {'pbeta': res.pval_nb_pbeta, 'pacat': res.pval_nb_pacat}
        df_pval = pd.DataFrame(data=d_pval)
        df_pval.to_csv(args.out + ".cisp.tsv", sep="\t", index=False)

    df_rej = pd.DataFrame(data=d_rej)
    df_rej.to_csv(args.out + ".tsv", sep="\t", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
