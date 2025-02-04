import pandas as pd

from pandas_plink import read_plink

from jax import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.sim_sc import run_sim


config.update("jax_enable_x64", True)

# Fix Va = 0.05, Vre=0.2:
# beta0 -> bulk_mean_ct/bulk coverage:

# CD4_NC (real data) bulk_mean_ct, after filter express_percent>10%:
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
#      0.10      0.92      6.31     96.32     23.92 108276.80
# -16 -> 0.2/17%; -14 -> 1.5/70%; -12.5 -> 7/90%; -11.5 -> 20/100%; -10 -> 90/100%; -7 -> 1700;  -5 -> 12k;

# B_IN
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
#     0.107     0.582     1.832    17.252     5.154 14029.689
# -16 -> 0.2/3%; -15 -> 0.8/8%; -12 -> 1.7/65%; -11 -> 4.5/90%; -10 -> 12/~100%; -5 -> 1800; -3 -> 13k

# Plasa:
#      Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
#    0.1082    0.3346    0.7572    4.9499    1.9660 1687.0377
# -13 -> 0.12/10%; -12 -> 0.3/20%; -11.5 -> 0.6/40%; -10 -> 2.2/70%; -9 -> 6/90%; -5 -> 300; -3 -> 2k

seed = 1
n = 1000
V_a = 0.05  # 0 for null
V_re = 0.2
beta0 = -15
maf = 0.2

# sample from observed
onek1k = pd.read_csv("../example/local/onek1k_3CT_sc/CD4_NC.features.tsv.gz", sep="\t")
sample_covar = onek1k[['individual', 'age', 'sex', 'iid_offset']]
sample_covar = sample_covar.drop_duplicates(subset=['individual', 'age', 'sex'], keep='last').reset_index(drop=True)

bim, fam, bed = read_plink("../example/local/onek1k_3CT_sc/chr1_0.2", verbose=False)
G = bed.compute()  # pxn array
snp = G[0].reshape(-1, 1)

# X, y, beta, sim_libsize, h2obs = sim_data(
#     libsize=onek1k, g=snp,
#     nobs=n, family=Poisson(), maf=maf,
#     V_a=V_a, V_re=V_re, beta0=beta0,
#     seed=seed, sample_covar=sample_covar
# )

res = run_sim(
    seed=1,
    family=Poisson(),
    bim=bim,
    bed=bed,
    nobs=n,
    onek1k_libsize=onek1k,
    maf=maf,
    beta0=beta0,  # intercept determine baseline counts
    V_a=V_a,
    V_re=V_re,
    m_causal=1,
    G=G,  # shape of num_sim x n
    sample_covar=sample_covar,  # nxp
    num_sim=3,
    out_path="../example/local/onek1k_3CT_sc/sim1",  # write out single cell data in saigeqtl format
)

print(res.bulk_mean_ct)
print(res.bulk_express_percent)
print(res.pval_nb_score)
print(res.pval_pois_score)
print(res.pval_lm_score)
print(res.alpha)
print('finish')
