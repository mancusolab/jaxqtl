import pandas as pd

from pandas_plink import read_plink

from jax import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.sim_sc import run_sim


config.update("jax_enable_x64", True)

seed = 1
n = 1000
V_a = 0  # 0 for null
V_re = 0.2
beta0 = -10
maf = 0.2

# sample from observed
onek1k = pd.read_csv("../example/local/onek1k_3CT_sc/CD4_NC.features.tsv.gz", sep="\t")
sample_covar = onek1k[['individual', 'age', 'sex', 'iid_offset']]
sample_covar = sample_covar.drop_duplicates(subset=['individual', 'age', 'sex'], keep='last').reset_index(drop=True)

bim, fam, bed = read_plink("../example/local/onek1k_3CT_sc/chr1_0.2", verbose=False)
G = bed.compute()  # pxn array
snp = G[0].reshape(-1, 1)

res = run_sim(
    seed=5,
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
    LD_path="../example/local/NK_new/ENSG00000156804.ld.raw.tsv.gz",
    sample_covar=sample_covar,  # nxp
    num_sim=5,
    out_path="../example/local/onek1k_3CT_sc/sim1_pbeta_acat",  # write out single cell data in saigeqtl format
)

print(res.bulk_mean_ct)
print(res.bulk_express_percent)
print(res.pval_nb_score)
print(res.pval_pois_score)
print(res.pval_lm_score)
print(res.alpha)
print('finish')
