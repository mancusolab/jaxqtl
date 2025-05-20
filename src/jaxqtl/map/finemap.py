from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import posie as po

import jax.random as rdm

from jax import numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from ..families.distribution import NegativeBinomial
from ..infer.glm import GLM
from ..infer.utils import CisGLMState
from ..io.readfile import ReadyDataState
from ..log import get_log
from .utils import _setup_G_y


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


def map_finemap(
    dat: ReadyDataState,
    out_path: str,
    standardize: bool = True,
    seed: int = 123,
    set_L: int = 10,
    step_size: float = 0.05,
    window: int = 500000,
    offset_eta: ArrayLike = 0.0,
    max_iter: int = 50,
    max_select: int = 100,
    verbose: bool = True,
    covar: Optional[ArrayLike] = None,
    log=None,
):
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
    n, _ = dat.pheno.count.shape

    gene_info = dat.pheno_meta

    # append genotype as the last column
    if (covar is not None) and standardize:
        _, k = covar.shape
        covar = (covar - jnp.mean(covar, axis=0)) / jnp.std(covar, axis=0)

    offset_eta = offset_eta.squeeze()

    key = rdm.PRNGKey(seed)

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
                "Performing fine mapping for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        y = y.squeeze()
        N, P = G.shape
        G = (G - jnp.mean(G, axis=0)) / jnp.std(G, axis=0)

        # set parameters
        set_L = int(jnp.array([set_L, P]).min())
        ss_b = (0.1 / set_L) * jnp.ones(set_L)
        pi = jnp.ones(P) / P

        # fit covar-only NB model to calculate dispersion
        jaxqtl_nb = GLM(family=NegativeBinomial())
        Z = jnp.concatenate([jnp.ones((N, 1)), covar], axis=1)
        init_eta, disp = jaxqtl_nb.calc_eta_and_dispersion(Z, y[:, jnp.newaxis], offset_eta[:, jnp.newaxis])
        glm_state = jaxqtl_nb.fit(
            Z,
            y[:, jnp.newaxis],
            init=init_eta,
            offset_eta=offset_eta[:, jnp.newaxis],
            alpha_init=disp.squeeze(),
        )
        ssu = jnp.log(glm_state.alpha + 1.0)

        result = po.infer(
            X=G,
            y=y,
            covar=covar,
            offset=offset_eta,
            L=set_L,
            pi=pi,
            init="prior",
            sigma_sq_b=ss_b,
            sigma_sq_u=ssu,
            step_size=step_size,
            max_iter=max_iter,
            max_select=max_select,
            threshold=0.95,
            purity_cutoff=0.5,
            seed=seed,
            optim="rg",
        )

        if verbose:
            log.info(
                "Finished cis-qtl scan for %s over region %s:%s-%s",
                gene_name,
                str(chrom),
                str(lstart),
                str(rend),
            )

        result_out = _prepare_finemap_result(result, gene_name, var_df)

        # write out results
        result_out.to_csv(f"{out_path}/{gene_name}.posie.tsv.gz", sep="\t", index=False)


def _prepare_finemap_result(
    result,
    gene_name: str,
    var_df: pd.DataFrame,
) -> pd.DataFrame:
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
    # note: the SNPIndex is 0-indexing
    # snp     chrom   pos     pip     cs1  cs2  ..
    # order by PIP so that easier for eye balling

    result_out = var_df.iloc[:, [0, 1, 3]].reset_index(drop=True)
    result_out['pip'] = result.pip
    result_out['phenotype_id'] = gene_name

    # re-order columns
    new_order = ['phenotype_id'] + [col for col in result_out.columns if col not in ['phenotype_id']]
    result_out = result_out[new_order]

    P, _ = var_df.shape

    cs = result.cs

    # write cs if there is any
    if len(cs) > 0:
        cs_list = result.cs['CSIndex'].unique()

        # create CS indicator columns
        for cs_idx in cs_list:
            new_col = f"cs{cs_idx}"
            result_out[new_col] = jnp.zeros((P,))
            indices_to_set = cs.loc[cs['CSIndex'] == cs_idx, :]['SNPIndex'].values.tolist()
            result_out.loc[indices_to_set, new_col] = 1

    result_out['converged'] = result.converged
    result_out['niter'] = result.niter
    result_out['elbo'] = result.elbo

    result_out.sort_values(by='pip', ascending=False, inplace=True)

    return result_out
