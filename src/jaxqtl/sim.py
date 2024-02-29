import argparse as ap
import logging
import sys

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import qtl.norm

from pandas_plink import read_plink

import jax
import jax.numpy as jnp
import jax.random as rdm

from jaxtyping import Array, ArrayLike

from jaxqtl.families.distribution import (
    ExponentialFamily,
    Gaussian,
    NegativeBinomial,
    Poisson,
)
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.stderr import FisherInfoError, HuberError
from jaxqtl.infer.utils import score_test_snp
from jaxqtl.log import get_logger


class SimState(NamedTuple):
    X: Array  # design matrix contains intercept + covariates + genotype
    y: Array
    beta: Array  # true betas
    libsize: Array
    h2obs: Array


class SimResState(NamedTuple):
    pval_nb_wald: Array
    pval_pois_wald: Array
    pval_nb_wald_robust: Array
    pval_pois_wald_robust: Array
    pval_lm_score: Array
    pval_lm_wald: Array
    pval_lm_wald_robust: Array
    pval_nb_score: Array
    pval_pois_score: Array


def sim_data(
    nobs: int = 1000,
    num_cells: int = 200,
    family: ExponentialFamily = Poisson(),
    method: str = "bulk",
    scale: float = 1.0,  # for linear model
    alpha: float = 0.0,  # for NB model
    maf: float = 0.3,
    beta0: float = 1.0,  # intercept determine baseline counts
    seed: int = 1,
    V_a: float = 0.1,
    V_re: float = 0.2,
    V_disp: float = 0.0,
    m_causal: int = 10,
    baseline_mu: float = 0.0,
    libsize: ArrayLike = 1,  # shape nx1 (only simulate per-individual offset, not cell level)
    geno_arr: Optional[ArrayLike] = None,
    sample_covar_arr: Optional[ArrayLike] = None,  # nxp, saigeqtl can't estimate ratio without covariates
) -> SimState:
    p = 2  # if covar not specified, then only intercept + genotype
    X = jnp.ones((nobs, 1))  # intercept

    key = rdm.PRNGKey(seed)

    # append sample level covariates to benchmark with saigeqtl
    if sample_covar_arr is not None:
        num_covar = sample_covar_arr.shape[1]
        p = num_covar + 2  # intercept + covars + genotype
        X = jnp.column_stack((X, sample_covar_arr))

    beta_shape = (p, 1)
    beta = jnp.ones(beta_shape)
    beta = beta.at[0].set(beta0)

    # simulate effect from age and sex
    if sample_covar_arr is not None:
        key, covar_key = rdm.split(key, 2)
        beta_covar = jnp.ones((num_covar, 1)) * 0.001  # fix covariate effect to be small
        beta = beta.at[1 : p - 1].set(beta_covar)

    # geno in shape of nx1
    if geno_arr is not None:
        g = geno_arr
    else:
        key, snp_key = rdm.split(key, 2)
        g = rdm.binomial(snp_key, 2, maf, shape=(nobs, 1))  # genotype (0,1,2)

    X = jnp.column_stack((X, g))  # include genotype at last column

    key, g_key = rdm.split(key, 2)
    g_beta = rdm.normal(g_key) * np.sqrt(V_a / m_causal) if V_a > 0 else 0.0

    # rescale to match specified V_a (!don't need this right now)
    # s2g = jnp.var(g * g_beta)
    # g_beta = g_beta * jnp.sqrt(V_a / s2g)

    beta = beta.at[-1].set(g_beta)  # put genotype as last column
    eta = X @ beta + jnp.log(libsize)

    if method == "bulk":
        mu = family.glink.inverse(eta)
        y = family.random_gen(mu, scale=scale, alpha=alpha)
        h2obs = jnp.array([-9])  # placeholder
    elif method == "sc":
        # sample random effect of each individual
        key, re_key = rdm.split(key, 2)
        bi = rdm.normal(re_key, (nobs, 1)) * np.sqrt(V_re) if V_re > 0 else 0
        eta = eta + bi
        mu = family.glink.inverse(eta)

        # for each individual mu_i, broadcast to num_cells
        if family == Poisson():
            key, y_key = rdm.split(key, 2)
            y = rdm.poisson(y_key, mu, shape=(nobs, num_cells))  # n x num_cells
        else:
            print("Only support Poisson() for single cell model")

        h2obs = _calc_h2obs(V_a, V_disp, V_re, baseline_mu)
    else:
        print("Specify either bulk or sc")

    return SimState(jnp.array(X), jnp.array(y), jnp.array(beta), jnp.array(libsize), h2obs)


def _calc_h2obs(V_a: float, V_disp: float, V_re: float, baseline_mu: float) -> Array:
    # Calculate heritability of additive genetics on observed scale
    tot_var = V_a + V_re + V_disp
    lamb = np.exp(baseline_mu + tot_var / 2.0)
    h2g_obs = lamb * V_a / (lamb * (np.exp(tot_var) - 1) + 1)
    return jnp.array(h2g_obs)


def run_sim(
    seed: int = 1,
    nobs: int = 1000,
    num_cells: int = 200,
    family: ExponentialFamily = Poisson(),
    method: str = "bulk",
    scale: float = 1.0,  # for linear model
    alpha: float = 0.0,  # for NB model
    maf: float = 0.3,
    beta0: float = 1.0,  # intercept determine baseline counts
    V_a: float = 0.1,
    V_re: float = 0.2,
    V_disp: float = 0.0,
    m_causal: int = 10,
    baseline_mu: float = 0.0,
    libsize: ArrayLike = 1,  # shape nx1 (only simulate per-individual offset)
    G: Optional[ArrayLike] = None,  # shape of num_sim x n
    sample_covar_arr: Optional[ArrayLike] = None,  # nxp
    num_sim: int = 1000,
    out_path: Optional[str] = None,  # write out single cell data in saigeqtl format
) -> SimResState:
    pval_nb_wald = jnp.array([])
    pval_nb_wald_robust = jnp.array([])
    pval_nb_score = jnp.array([])

    pval_pois_wald = jnp.array([])
    pval_pois_wald_robust = jnp.array([])
    pval_pois_score = jnp.array([])

    pval_lm_wald = jnp.array([])
    pval_lm_wald_robust = jnp.array([])
    pval_lm_score = jnp.array([])

    for i in range(num_sim):
        snp = None if G is None else G[i].reshape(-1, 1)
        X, y, beta, libsize, h2obs = sim_data(
            nobs=nobs,
            num_cells=num_cells,
            family=family,
            method=method,
            scale=scale,  # for linear model
            alpha=alpha,  # for NB model
            maf=maf,
            beta0=beta0,  # intercept determine baseline counts
            seed=i + seed,  # use simulation index
            V_a=V_a,
            V_re=V_re,
            V_disp=V_disp,
            m_causal=m_causal,
            baseline_mu=baseline_mu,
            libsize=libsize,  # shape nx1 (only simulate per-individual offset)
            geno_arr=snp,  # genotype is generated for num_sim
            sample_covar_arr=sample_covar_arr,  # nxp
        )

        if method == "sc":
            log_offset = jnp.repeat(jnp.log(libsize), num_cells)  # repeat each element n times [1,2,3] -> [1,1,2,2,3,3]
            y_mat = jnp.column_stack((log_offset.reshape(-1, 1), y.ravel().reshape(-1, 1)))
            df = pd.DataFrame(y_mat).reset_index()
            df.columns = ['iid', 'log_offset', 'gene' + str(i + 1)]

            iid_index = jnp.arange(1, nobs + 1)
            df.iid = jnp.repeat(iid_index, num_cells)

            if sample_covar_arr is not None:
                df['sex'] = jnp.repeat(X[:, 1], num_cells)
                df['age'] = jnp.repeat(X[:, 2], num_cells)
            df.to_csv(f"{out_path}.pheno{i+1}.tsv.gz", sep="\t", index=False)

            # convert back to pseudo-bulk for jaxqtl
            y = y.sum(axis=1).reshape(-1, 1)
            pd.DataFrame({'mean_ct': [y.mean()]}).to_csv(
                f"{out_path}.pheno{i+1}.mean_pseudo_ct.tsv.gz", sep="\t", index=False
            )

        log_offset = jnp.log(libsize)
        jaxqtl_pois = GLM(family=Poisson())
        jaxqtl_nb = GLM(family=NegativeBinomial())
        jaxqtl_lm = GLM(family=Gaussian())

        # fit poisson wald test
        init_pois = jaxqtl_pois.family.init_eta(y)
        glm_state_pois = jaxqtl_pois.fit(X, y, init=init_pois, offset_eta=log_offset, se_estimator=FisherInfoError())

        pval_pois_wald = jnp.append(pval_pois_wald, glm_state_pois.p[-1])

        # fit NB wald test
        init_nb, alpha_n = jaxqtl_nb.calc_eta_and_dispersion(X, y, log_offset)
        alpha_n = jnp.nan_to_num(alpha_n, nan=0.1)

        glm_state_nb = jaxqtl_nb.fit(
            X, y, init=init_nb, alpha_init=alpha_n, offset_eta=log_offset, se_estimator=FisherInfoError()
        )

        pval_nb_wald = jnp.append(pval_nb_wald, glm_state_nb.p[-1])

        # robust poisson and NB
        glm_state_pois = jaxqtl_pois.fit(X, y, init=init_pois, se_estimator=HuberError(), offset_eta=log_offset)
        glm_state_nb = jaxqtl_nb.fit(
            X, y, init=init_nb, alpha_init=alpha_n, offset_eta=log_offset, se_estimator=HuberError()
        )

        pval_pois_wald_robust = jnp.append(pval_pois_wald_robust, glm_state_pois.p[-1])
        pval_nb_wald_robust = jnp.append(pval_nb_wald_robust, glm_state_nb.p[-1])

        # fit lm (genexN); only one gene so don't need convert to cpm per individual
        norm_df = qtl.norm.inverse_normal_transform(pd.DataFrame(y).T)
        y_norm = np.array(norm_df.T)

        init_lm = jaxqtl_lm.family.init_eta(y_norm)
        glm_state = jaxqtl_lm.fit(X, y_norm, init=init_lm, se_estimator=FisherInfoError())

        pval_lm_wald = jnp.append(pval_lm_wald, glm_state.p[-1])

        glm_state = jaxqtl_lm.fit(X, y_norm, init=init_lm, se_estimator=HuberError())
        pval_lm_wald_robust = jnp.append(pval_lm_wald_robust, glm_state.p[-1])

        # score test for poisson and NB
        X_cov = X[:, 0:-1]
        g = X[:, -1].reshape(-1, 1)

        glm_null_pois = jaxqtl_pois.fit(X_cov, y, init=init_pois, offset_eta=log_offset)
        _, pval, _, _ = score_test_snp(G=g, X=X_cov, glm_null_res=glm_null_pois)

        pval_pois_score = jnp.append(pval_pois_score, pval)

        init_nb, alpha_n = jaxqtl_nb.calc_eta_and_dispersion(X_cov, y, log_offset)
        alpha_n = jnp.nan_to_num(alpha_n, nan=0.1)

        glm_state_nb = jaxqtl_nb.fit(X_cov, y, init=init_nb, alpha_init=alpha_n, offset_eta=log_offset)
        _, pval, _, _ = score_test_snp(G=g, X=X_cov, glm_null_res=glm_state_nb)

        pval_nb_score = jnp.append(pval_nb_score, pval)

        glm_state_lm = jaxqtl_lm.fit(X_cov, y, init=init_lm)
        _, pval, _, _ = score_test_snp(G=g, X=X_cov, glm_null_res=glm_state_lm)

        pval_lm_score = jnp.append(pval_lm_score, pval)

    return SimResState(
        pval_nb_wald=pval_nb_wald,
        pval_nb_wald_robust=pval_nb_wald_robust,
        pval_nb_score=pval_nb_score,
        pval_pois_wald=pval_pois_wald,
        pval_pois_wald_robust=pval_pois_wald_robust,
        pval_pois_score=pval_pois_score,
        pval_lm_wald=pval_lm_wald,
        pval_lm_wald_robust=pval_lm_wald_robust,
        pval_lm_score=pval_lm_score,
    )


def main(args):
    argp = ap.ArgumentParser(description="")  # create an instance
    argp.add_argument("-geno", type=str, help="Genotype plink prefix, eg. chr17")
    argp.add_argument("-covar", type=str, help="Path to covariates, include age, sex and library size")
    argp.add_argument("-libsize-fix", type=int, help="fixed library size value")
    argp.add_argument("-nobs", type=int, help="Sample size")
    argp.add_argument("-num-cells", type=int, default=100, help="Number of cells per person")
    argp.add_argument("-m-causal", type=int, help="Number of causal variants")
    argp.add_argument("-model", type=str, choices=["gaussian", "poisson", "NB"], help="Model")
    argp.add_argument("-beta0", type=float, default=0, help="Intercept")
    argp.add_argument("-Va", type=float, default=0.1, help="Variance explained by genetics, 0 means eqtl_beta=0")
    argp.add_argument("-Vre", type=float, default=0.2, help="Variance explained by random effect (across individuals)")
    argp.add_argument("-Vdisp", type=float, default=0.0, help="Variance dispersion")
    argp.add_argument("-baseline-mu", type=float, default=0.0, help="population baseline mu on observed scale")
    argp.add_argument("-alpha", type=float, default=0.0, help="True dispersion parameter when simulating NB")
    argp.add_argument("-scale", type=float, default=0.0, help="used in linear model")
    argp.add_argument("-maf", type=float, default=0.1, help="MAF")
    argp.add_argument("-seed", type=int, default=1, help="seed")
    argp.add_argument("-num-sim", type=int, default=1000, help="Number of simulation, equal to #markers in plink file")
    argp.add_argument("-fwer", type=float, default=0.05)
    argp.add_argument("-method", type=str, choices=["bulk", "sc"], help="either bulk or sc")
    argp.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose for logger",
    )
    argp.add_argument("-out-sc", type=str, help="out file prefix for saigeqtl phenotype")
    argp.add_argument("-out", type=str, help="out file prefix")

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

    if args.covar is not None:
        covar_df = pd.read_csv(args.covar, sep="\t")
        covar = jnp.array(covar_df[['sex', 'age']])
        covar = covar / jnp.std(covar, axis=0)
        log.info("Read in covar file.")
    else:
        covar = None

    if args.libsize_fix is None:
        libsize = jnp.array(covar_df['libsize']).reshape((-1, 1))
        log.info("Use library size from covar file")
    else:
        libsize = jnp.ones((args.nobs, 1)) * args.libsize_fix
        log.info(f"Use fixed library size : {args.libsize_fix}")

    res = run_sim(
        nobs=args.nobs,
        num_cells=args.num_cells,
        family=family,
        method=args.method,
        scale=args.scale,  # for linear model
        alpha=args.alpha,  # for NB model
        maf=args.maf,
        beta0=args.beta0,  # intercept determine baseline counts
        V_a=args.Va,
        V_re=args.Vre,
        V_disp=args.Vdisp,
        m_causal=args.m_causal,
        baseline_mu=args.baseline_mu,
        libsize=libsize,  # shape nx1 (only simulate per-individual offset)
        G=G,
        seed=args.seed,
        sample_covar_arr=covar,  # nxp
        num_sim=args.num_sim,
        out_path=args.out_sc,  # write out single cell data in saigeqtl format
    )

    d = {
        'rej_nb_wald': [jnp.mean(res.pval_nb_wald[~jnp.isnan(res.pval_nb_wald)] < args.fwer)],
        'rej_nb_wald_robust': [jnp.mean(res.pval_nb_wald_robust[~jnp.isnan(res.pval_nb_wald_robust)] < args.fwer)],
        'rej_nb_score': [jnp.mean(res.pval_nb_score[~jnp.isnan(res.pval_nb_score)] < args.fwer)],
        'rej_pois_wald': [jnp.mean(res.pval_pois_wald[~jnp.isnan(res.pval_pois_wald)] < args.fwer)],
        'rej_pois_wald_robust': [
            jnp.mean(res.pval_pois_wald_robust[~jnp.isnan(res.pval_pois_wald_robust)] < args.fwer)
        ],
        'rej_pois_score': [jnp.mean(res.pval_pois_score[~jnp.isnan(res.pval_pois_score)] < args.fwer)],
        'rej_lm_wald': [jnp.mean(res.pval_lm_wald[~jnp.isnan(res.pval_lm_wald)] < args.fwer)],
        'rej_lm_wald_robust': [jnp.mean(res.pval_lm_wald_robust[~jnp.isnan(res.pval_lm_wald_robust)] < args.fwer)],
        'rej_lm_score': [jnp.mean(res.pval_lm_score[~jnp.isnan(res.pval_lm_score)] < args.fwer)],
    }

    df_rej = pd.DataFrame(data=d)
    df_rej.to_csv(args.out + ".tsv", sep="\t", index=False)

    # write out pvalues of eqtls
    d = {
        'rej_nb_wald': res.pval_nb_wald,
        'rej_nb_wald_robust': res.pval_nb_wald_robust,
        'rej_nb_score': res.pval_nb_score,
        'rej_pois_wald': res.pval_pois_wald,
        'rej_pois_wald_robust': res.pval_pois_wald_robust,
        'rej_pois_score': res.pval_pois_score,
        'rej_lm_wald': res.pval_lm_wald,
        'rej_lm_wald_robust': res.pval_lm_wald_robust,
        'rej_lm_score': res.pval_lm_score,
    }

    df_pval = pd.DataFrame(data=d)
    df_pval.to_csv(args.out + ".pval.tsv.gz", sep="\t", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
