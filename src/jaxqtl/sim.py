import argparse as ap
import logging
import sys

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import qtl.norm

from numpy import ndarray
from pandas_plink import read_plink

import jax.numpy as jnp
import jax.random as rdm

from jax import config
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
    X: Array
    y: Array
    beta: Array
    libsize: Array
    h2obs: Array


class SimResState(NamedTuple):
    pval_nb_wald: ndarray
    pval_pois_wald: ndarray
    pval_nb_wald_robust: ndarray
    pval_pois_wald_robust: ndarray
    pval_lm_score: ndarray
    pval_lm_wald: ndarray
    pval_lm_wald_robust: ndarray
    pval_nb_score: ndarray
    pval_pois_score: ndarray


def sim_data(
    nobs: int = 1000,
    family: ExponentialFamily = Poisson(),
    method: str = "bulk",
    scale: float = 1.0,
    alpha: float = 0.0,
    maf: float = 0.3,
    beta0: float = 1.0,
    eqtl_beta: Optional[float] = None,
    seed: int = 1,
    V_a: float = 0.1,
    V_re: float = 0.5,
    V_disp: float = 0.0,
    m_causal: int = 10,
    num_cells: int = 1000,
    baseline_mu: float = 0.0,
    libsize: ArrayLike = 10000,  # shape nx1 (only simulate per-individual offset)
    geno_arr: Optional[ArrayLike] = None,
    covar_var: float = 0.1,
    sample_covar_arr: Optional[ArrayLike] = None,  # nxp
) -> SimState:
    n = nobs
    p = 2  # for now only intercept + genotype
    X = jnp.ones((n, 1))  # intercept

    np.random.seed(seed)
    key = rdm.PRNGKey(seed)

    if sample_covar_arr is not None:
        num_covar = sample_covar_arr.shape[1]
        p = num_covar + 2  # covars + intercept + genotype
        X = np.column_stack((X, sample_covar_arr))

    beta_shape = (p, 1)
    beta = np.ones(beta_shape)
    beta[0] = beta0
    if sample_covar_arr is not None:
        key, covar_key = rdm.split(key, 2)
        beta[1 : p - 1] = rdm.normal(covar_key, size=(num_covar, 1)) * np.sqrt(covar_var)

    # geno in shape of nx1
    if geno_arr is not None:
        g = geno_arr
    else:
        g = np.random.binomial(2, maf, (n, 1))  # genotype (0,1,2)

    X = np.column_stack((X, g))  # include genotype as the last column

    if eqtl_beta is None:
        # sample eqtl effect from  N(0, V_a/M)
        key, g_key = rdm.split(key, 2)
        g_beta = rdm.normal(g_key) * np.sqrt(V_a / m_causal)
    else:
        g_beta = eqtl_beta

    beta[-1] = g_beta  # put genotype as last column
    eta = X @ beta + jnp.log(libsize)

    if method == "bulk":
        mu = family.glink.inverse(eta)
        y = family.random_gen(mu, scale=scale, alpha=alpha)
        h2obs = -9  # placeholder
    elif method == "sc":
        # sample random effect of each individual
        key, re_key = rdm.split(key, 2)
        bi = rdm.normal(re_key, (n, 1)) * np.sqrt(V_re)
        eta = eta + bi
        mu = family.glink.inverse(eta)

        # for each individual mu_i, broadcast to num_cells
        key, y_key = rdm.split(key, 2)
        if family == Poisson():
            y = rdm.poisson(y_key, mu, shape=(n, num_cells))  # n x num_cells
        else:
            print("Only support Poisson()")

        h2obs = _calc_h2obs(V_a, V_disp, V_re, baseline_mu)
    else:
        print("Specify either bulk or sc")

    return SimState(jnp.array(X), jnp.array(y), jnp.array(beta), jnp.array(libsize), h2obs)


def _calc_h2obs(V_a: float, V_disp: float, V_re: float, baseline_mu: float) -> Array:
    # Calculate heritability of additive genetics on liability scale
    tot_var = V_a + V_re + V_disp
    lamb = np.exp(baseline_mu + tot_var / 2.0)
    h2g_obs = lamb * V_a / (lamb * (np.exp(tot_var) - 1) + 1)
    return jnp.array(h2g_obs)


def run_sim(
    scale: float = 1.0,
    alpha: float = 0.0,
    maf: float = 0.3,
    n: int = 1000,
    m_causal: int = 10,
    num_sim: int = 1000,
    beta0: float = 1.0,
    eqtl_beta: Optional[float] = None,
    family: ExponentialFamily = NegativeBinomial(),
    libsize: ArrayLike = 1,
    num_cells: int = 100,
    method: str = "sc",
    geno_arr: Optional[Array] = None,  # marker x iid
    out_path: Optional[str] = None,
    sample_covar_arr: Optional[ArrayLike] = None,  # nxp
    covar_var: float = 0.1,
) -> SimResState:
    pval_nb_wald = np.array([])
    pval_nb_wald_robust = np.array([])
    pval_nb_score = np.array([])

    pval_pois_wald = np.array([])
    pval_pois_wald_robust = np.array([])
    pval_pois_score = np.array([])

    pval_lm_wald = np.array([])
    pval_lm_wald_robust = np.array([])
    pval_lm_score = np.array([])

    for i in range(1, num_sim + 1):
        X, y, beta, libsize, h2obs = sim_data(
            nobs=n,
            family=family,
            sample_covar_arr=sample_covar_arr,
            maf=maf,
            geno_arr=geno_arr,
            num_cells=num_cells,
            m_causal=m_causal,
            eqtl_beta=eqtl_beta,
            alpha=alpha,
            beta0=beta0,
            scale=scale,
            seed=i,
            method=method,
            libsize=libsize,
            covar_var=covar_var,
        )

        if method == "sc":
            log_offset = jnp.repeat(jnp.log(libsize), num_cells)
            y_mat = jnp.column_stack((log_offset.reshape(-1, 1), y.ravel().reshape(-1, 1)))
            df = pd.DataFrame(y_mat).reset_index()
            df.columns = ['iid', 'log_offset', 'gene' + str(i)]

            iid_index = jnp.arange(1, n + 1)
            df.iid = jnp.repeat(iid_index, num_cells)

            if sample_covar_arr is not None:
                df['sex'] = jnp.repeat(X[:, 1], num_cells)
                df['age'] = jnp.repeat(X[:, 2], num_cells)
            df.to_csv(f"{out_path}.pheno{i}.tsv.gz", sep="\t", index=False)

            # convert back to pseudo-bulk
            y = jnp.array(df.groupby('iid')['gene' + str(i)].sum()).reshape(-1, 1)
            log_offset = jnp.log(libsize)

        # fit poisson wald test
        jaxqtl_pois = GLM(family=Poisson())
        init_pois = jaxqtl_pois.family.init_eta(y)
        glm_state_pois = jaxqtl_pois.fit(X, y, init=init_pois, se_estimator=FisherInfoError())

        pval_pois_wald = np.append(pval_pois_wald, glm_state_pois.p[-1])

        # fit NB wald test
        jaxqtl_nb = GLM(family=NegativeBinomial())
        init_eta, alpha_n = jaxqtl_nb.calc_eta_and_dispersion(X, y, log_offset)
        alpha_n = jnp.nan_to_num(alpha_n, nan=0.1)

        glm_state_nb = jaxqtl_nb.fit(X, y, init=glm_state_pois.eta, alpha_init=alpha_n, se_estimator=FisherInfoError())

        pval_nb_wald = np.append(pval_nb_wald, glm_state_nb.p[-1])

        # robust poisson and NB
        glm_state_pois = jaxqtl_pois.fit(X, y, init=init_pois, se_estimator=HuberError())
        glm_state_nb = jaxqtl_nb.fit(X, y, init=glm_state_pois.eta, alpha_init=alpha_n, se_estimator=HuberError())

        pval_pois_wald_robust = np.append(pval_pois_wald_robust, glm_state_pois.p[-1])
        pval_nb_wald_robust = np.append(pval_nb_wald_robust, glm_state_nb.p[-1])

        # fit lm
        norm_df = qtl.norm.inverse_normal_transform(pd.DataFrame(y).T)
        y_norm = np.array(norm_df.T)

        jaxqtl_lm = GLM(family=Gaussian())
        init_lm = jaxqtl_lm.family.init_eta(y_norm)
        glm_state = jaxqtl_lm.fit(X, y_norm, init=init_lm, se_estimator=FisherInfoError())

        pval_lm_wald = np.append(pval_lm_wald, glm_state.p[-1])

        glm_state = jaxqtl_lm.fit(X, y_norm, init=init_lm, se_estimator=HuberError())
        pval_lm_wald_robust = np.append(pval_lm_wald_robust, glm_state.p[-1])

        # score test for poisson and NB
        X_cov = X[:, 0:-1]
        glm_null_pois = jaxqtl_pois.fit(X_cov, y, init=init_pois)
        _, pval, _, _ = score_test_snp(G=X[:, -1].reshape(-1, 1), X=X_cov, glm_null_res=glm_null_pois)

        pval_pois_score = np.append(pval_pois_score, pval)

        init_eta, alpha_n = jaxqtl_nb.calc_eta_and_dispersion(X, y, log_offset)
        alpha_n = jnp.nan_to_num(alpha_n, nan=0.1)

        glm_state_nb = jaxqtl_nb.fit(X_cov, y, init=glm_null_pois.eta, alpha_init=alpha_n)
        _, pval, _, _ = score_test_snp(G=X[:, -1].reshape((n, 1)), X=X_cov, glm_null_res=glm_state_nb)

        pval_nb_score = np.append(pval_nb_score, pval)

        glm_state_lm = jaxqtl_lm.fit(X_cov, y, init=init_lm)
        _, pval, _, _ = score_test_snp(G=X[:, -1].reshape((n, 1)), X=X_cov, glm_null_res=glm_state_lm)

        pval_lm_score = np.append(pval_lm_score, pval)

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
    argp.add_argument("-geno", type=str, help="Genotype prefix, eg. chr17")
    argp.add_argument("-n", type=int, help="Sample size")
    argp.add_argument("-m", type=int, help="Number of causal variants")
    argp.add_argument("-model", type=str, choices=["gaussian", "poisson", "NB"], help="Model")
    argp.add_argument("-libsize", type=str, help="Path to library size (no header)")
    argp.add_argument("-alpha", type=float, default=0.05, help="True dispersion parameter when simulating NB")
    argp.add_argument("-maf", type=float, default=0.1, help="MAF")
    argp.add_argument("-test-method", type=str, choices=["wald", "score"], help="Wald or score test")
    argp.add_argument("-window", type=int, default=500000)
    argp.add_argument("-fwer", type=float, default=0.05)
    argp.add_argument("-g-index", type=int, default=1, help="index of causal SNP (1-based)")
    argp.add_argument("-method", type=str, choices=["bulk", "sc"], help="either bulk or sc")
    argp.add_argument("--seed", type=int, default=1)
    argp.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose for logger",
    )
    argp.add_argument("-out", type=str, help="out file prefix")

    args = argp.parse_args(args)  # a list a strings

    platform = "cpu"
    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", platform)

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
        snp = G[args.g_index - 1]  # change to 0-based

        res = run_sim(
            n=args.n,
            family=family,
            geno_arr=snp[:, np.newaxis],
            alpha=args.alpha,
            maf=args.maf,
            model="alt",
            eqtl_beta=args.true_beta,
            seed=args.seed,
            libsize=args.libsize,
        )

    else:
        # simulate g for one SNP
        res = run_sim(n=args.n, family=family, alpha=0.0, maf=0.3, eqtl_beta=args.true_beta, seed=args.seed, libsize=1)

    d = {
        'rej_nb_wald': [np.mean(res.pval_nb_wald[~np.isnan(res.pval_nb_wald)] < args.fwer)],
        'rej_nb_wald_robust': [np.mean(res.pval_nb_wald_robust[~np.isnan(res.pval_nb_wald_robust)] < args.fwer)],
        'rej_nb_score': [np.mean(res.pval_nb_score[~np.isnan(res.pval_nb_score)] < args.fwer)],
        'rej_pois_wald': [np.mean(res.pval_pois_wald[~np.isnan(res.pval_pois_wald)] < args.fwer)],
        'rej_pois_wald_robust': [np.mean(res.pval_pois_wald_robust[~np.isnan(res.pval_pois_wald_robust)] < args.fwer)],
        'rej_pois_score': [np.mean(res.pval_pois_score[~np.isnan(res.pval_pois_score)] < args.fwer)],
        'rej_lm_wald': [np.mean(res.pval_lm_wald[~np.isnan(res.pval_lm_wald)] < args.fwer)],
        'rej_lm_wald_robust': [np.mean(res.pval_lm_wald_robust[~np.isnan(res.pval_lm_wald_robust)] < args.fwer)],
        'rej_lm_score': [np.mean(res.pval_lm_score[~np.isnan(res.pval_lm_score)] < args.fwer)],
    }

    df_rej = pd.DataFrame(data=d)
    df_rej.to_csv(args.out + ".tsv", sep="\t", index=False)

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
    df_pval.to_csv(args.out + ".pval.tsv", sep="\t", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
