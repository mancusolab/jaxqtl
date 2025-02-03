import pandas as pd

from pandas_plink import read_plink

from jax import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.sim_sc import run_sim


config.update("jax_enable_x64", True)

# Fix Va = 0.05, Vre=0.2:
# beta0 -> bulk coverage:
# -17 -> 8%; -16.5 -> 10%; -16 -> 18%; -15 -> 40%; -14.5 -> 55%; -14 -> 70%; -13.5: 80%; -13 -> 90%; -12 -> ~100%

seed = 3
n = 1000
V_a = 0.05  # 0 for null
V_re = 0.2
beta0 = -16.5
maf = 0.2

# sample from observed
onek1k = pd.read_csv("../example/local/onek1k_3CT_sc/CD4_NC.features.tsv.gz", sep="\t")
sample_covar = onek1k.drop_duplicates(subset=['individual', 'age', 'sex'], keep='last').reset_index(drop=True)

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
    num_sim=5,
    out_path="../example/local/onek1k_3CT_sc/sim1",  # write out single cell data in saigeqtl format
)

print('finish')
