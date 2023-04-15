# Adapt from code in tensorqtl created by Francois Aguet

import os
import subprocess
import sys

import pandas as pd
import scipy.stats as stats

sys.path.insert(1, os.path.dirname(__file__))  # ?


has_rpy2 = False
e = subprocess.call(
    "which R", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
try:
    from jaxqtl.post import rfunc

    if e == 0:
        has_rpy2 = True
except ImportWarning:  # ? not sure if use this type of error
    pass
if not has_rpy2:
    print(
        "Warning: 'rfunc' cannot be imported. R and the 'rpy2' Python package are needed."
    )


def calculate_qvalues(
    cis_df: pd.DataFrame, log, fdr: float = 0.05, qvalue_lambda=None
) -> pd.DataFrame:
    """Annotate permutation results with q-values, p-value threshold"""

    log.info("Computing q-values")
    log.info(f"  * Number of phenotypes tested: {cis_df.shape[0]}")

    if not cis_df["pval_beta"].isnull().all():
        pval_col = "pval_beta"
        r = stats.pearsonr(cis_df["pval_perm"], cis_df["pval_beta"])[
            0
        ]  # correlation statistics
        log.info(
            f"  * Correlation between Beta-approximated and empirical p-values: {r:.4f}"
        )
    else:
        pval_col = "pval_perm"
        log.info(
            "  * WARNING: no beta-approximated p-values found, using permutation p-values instead."
        )

    # calculate q-values
    if qvalue_lambda is not None:
        log.info(f"  * Calculating q-values with lambda = {qvalue_lambda:.3f}")
    qval, pi0 = rfunc.qvalue(cis_df[pval_col], lambda_qvalue=qvalue_lambda)

    cis_df["qval"] = qval
    log.info(f"  * Proportion of significant phenotypes (1-pi0): {1-pi0:.2f}")
    log.info(f"  * QTL phenotypes @ FDR {fdr:.2f}: {(cis_df['qval'] <= fdr).sum()}")

    # determine global min(p) significance threshold and calculate nominal p-value threshold for each gene
    if pval_col == "pval_beta":
        lb = cis_df.loc[cis_df["qval"] <= fdr, "pval_beta"].sort_values()  # ascending
        ub = cis_df.loc[cis_df["qval"] > fdr, "pval_beta"].sort_values()

        if lb.shape[0] > 0:  # significant phenotypes
            lb = lb[-1]
            if ub.shape[0] > 0:
                ub = ub[0]
                pthreshold = (lb + ub) / 2
            else:
                pthreshold = lb
            log.info(f"  * min p-value threshold @ FDR {fdr}: {pthreshold:.6g}")
            cis_df["pval_nominal_threshold"] = stats.beta.ppf(
                pthreshold, cis_df["beta_shape1"], cis_df["beta_shape2"]
            )

    return cis_df
