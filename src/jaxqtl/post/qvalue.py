# import numpy as np
# import pandas as pd
# import scipy.stats as stats
# import subprocess
# import sys
# import os
# from jaxqtl.log import get_log
# import glob
# from datetime import datetime
#
# sys.path.insert(1, os.path.dirname(__file__))
# from core import SimpleLogger, Residualizer, center_normalize, impute_mean, get_allele_stats
# import mixqtl
# import qtl.genotype as gt
#
#
# has_rpy2 = False
# e = subprocess.call('which R', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# try:
#     import rpy2
#     import rfunc
#     if e == 0:
#         has_rpy2 = True
# except:
#     pass
# if not has_rpy2:
#     print("Warning: 'rfunc' cannot be imported. R and the 'rpy2' Python package are needed.")
#
#
# def calculate_qvalues(cis_df, fdr=0.05, qvalue_lambda=None, logger=None):
#     """Annotate permutation results with q-values, p-value threshold"""
#     log = get_log()
#
#     log.info('Computing q-values')
#     log.info(f'  * Number of phenotypes tested: {cis_df.shape[0]}')
#
#     if not cis_df['pval_beta'].isnull().all():
#         pval_col = 'pval_beta'
#         r = stats.pearsonr(cis_df['pval_perm'], cis_df['pval_beta'])[0]
#         log.info(f'  * Correlation between Beta-approximated and empirical p-values: {r:.4f}')
#     else:
#         pval_col = 'pval_perm'
#         log.info(f'  * WARNING: no beta-approximated p-values found, using permutation p-values instead.')
#
#     # calculate q-values
#     if qvalue_lambda is not None:
#         log.info(f'  * Calculating q-values with lambda = {qvalue_lambda:.3f}')
#     qval, pi0 = rfunc.qvalue(cis_df[pval_col], lambda_qvalue=qvalue_lambda)
#
#     cis_df['qval'] = qval
#     logger.write(f'  * Proportion of significant phenotypes (1-pi0): {1-pi0:.2f}')
#     logger.write(f"  * QTL phenotypes @ FDR {fdr:.2f}: {(cis_df['qval'] <= fdr).sum()}")
#
#     # determine global min(p) significance threshold and calculate nominal p-value threshold for each gene
#     if pval_col == 'pval_beta':
#         lb = cis_df.loc[cis_df['qval'] <= fdr, 'pval_beta'].sort_values()
#         ub = cis_df.loc[cis_df['qval'] > fdr, 'pval_beta'].sort_values()
#
#         if lb.shape[0] > 0:  # significant phenotypes
#             lb = lb[-1]
#             if ub.shape[0] > 0:
#                 ub = ub[0]
#                 pthreshold = (lb+ub)/2
#             else:
#                 pthreshold = lb
#             logger.write(f'  * min p-value threshold @ FDR {fdr}: {pthreshold:.6g}')
#             cis_df['pval_nominal_threshold'] = stats.beta.ppf(pthreshold, cis_df['beta_shape1'],
#                                                               cis_df['beta_shape2'])
