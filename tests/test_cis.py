import timeit

import pandas as pd

from utils import assert_array_eq

import jax.numpy as jnp

from jax import config

from jaxqtl.families.distribution import Gaussian, NegativeBinomial, Poisson
from jaxqtl.infer.utils import ScoreTest, WaldTest  # , RareTest
from jaxqtl.io.covar import covar_reader
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import PheBedReader
from jaxqtl.io.readfile import create_readydata
from jaxqtl.log import get_log
from jaxqtl.map.cis import map_cis, write_parqet
from jaxqtl.map.nominal import map_nominal


pd.set_option("display.max_columns", 500)  # see cis output

config.update("jax_enable_x64", True)

geno_path = "../example/local/NK_new/chr22"
covar_path = "../example/local/NK_new/donor_features.all.6PC.tsv"
pheno_path = "../example/local/NK_new/NK.chr22.bed.gz"
# genelist_path = "../example/local/NK_new/ENSG00000198125"
genelist_path = "../example/data/genelist_spatest.tsv"
# offset_path = "../example/local/NK_new/log_libsize.tsv"
offset_path = None
log = get_log()

# raw genotype data and impute for genotype data
log.info("Load genotype.")
geno_reader = PlinkReader()
geno, bim, sample_info = geno_reader(geno_path)

log.info("Load covariates.")
# covar = covar_reader(covar_path, addcovar_path, covar_test)
covar = covar_reader(covar_path)

log.info("Load phenotype.")
pheno_reader = PheBedReader()
pheno = pheno_reader(pheno_path)

# run Mapping #
dat = create_readydata(geno, bim, pheno, covar, autosomal_only=True)

maf_threshold = 0.0
dat.filter_geno(maf_threshold, "22")

# add phenotype PCs
dat.filter_gene(geneexpr_percent_cutoff=0.0)

dat.add_covar_pheno_PC(k=2, add_covar=None)

# filter phenotype (5 genes)
gene_list = pd.read_csv(genelist_path, sep="\t")["phenotype_id"].to_list()

# before filter gene list, calculate library size and set offset
if offset_path is None:
    total_libsize = jnp.array(dat.pheno.count.sum(axis=1))[:, jnp.newaxis]
    offset_eta = jnp.log(total_libsize)
else:
    offset_eta = pd.read_csv("../example/local/NK_new/log_libsize.tsv", names=['iid', 'eta'], sep="\t", index_col="iid")
    offset_eta = offset_eta.loc[offset_eta.index.isin(dat.pheno.count.index)].sort_index()
    offset_eta = jnp.array(offset_eta)

# dat.filter_gene(gene_list=[gene_list[0]])  # filter to one gene
dat.filter_gene(gene_list=["ENSG00000169575"])

# run mapping #

# read Rres for score test and wald test
R_res = pd.read_csv("../example/data/n94_wald_scoretest_pois_Rres.tsv", sep="\t")


# score test
def test_cis_scoretest():
    """
    use data of N=94
    """
    start = timeit.default_timer()

    outdf = map_nominal(dat, family=Poisson(), offset_eta=offset_eta, test=ScoreTest())
    stop = timeit.default_timer()
    print("Time: ", stop - start)
    write_parqet(outdf=outdf, method="scoretest", out_path="../example/result/dat_n94_test")

    # Z lose accuracy due to divisions from recalculations
    assert_array_eq(outdf.slope / outdf.slope_se, jnp.array(R_res["Z_scoretest"]), rtol=5e-4)
    assert_array_eq(outdf.pval_nominal, jnp.array(R_res["pval_scoretest"]))


# Wald test and slope, se estimates
def test_cis_waldtest():
    """
    use data of N=94
    """
    # n=94, one gene nominal mapping, 2592 variants, 916 ms
    # 0.86s vs. 0.90
    start = timeit.default_timer()
    outdf = map_nominal(dat, family=Poisson(), offset_eta=offset_eta, robust_se=False, test=WaldTest())
    stop = timeit.default_timer()
    print("Time: ", stop - start)
    write_parqet(outdf=outdf, method="wald", out_path="../example/result/dat_n94_test")

    assert_array_eq(outdf.slope, jnp.array(R_res["slope"]))
    assert_array_eq(outdf.slope_se, jnp.array(R_res["se"]))


# map_intercept = fit_intercept_only(dat, family=Poisson(), offset_eta=offset_eta, robust_se=False)

# ~4s
start = timeit.default_timer()
mapcis_out_score_nb = map_cis(
    dat,
    family=NegativeBinomial(),
    test=ScoreTest(),
    offset_eta=offset_eta,
    n_perm=200,
    compute_qvalue=False,
)
stop = timeit.default_timer()
print("Time: ", stop - start)
# mapcis_out_score_nb.to_csv("../example/result/n94_scoretest_NB_res.tsv", sep="\t", index=False)


out_nb = map_nominal(dat, family=NegativeBinomial(), offset_eta=offset_eta, test=ScoreTest(), robust_se=True)
# # out_nb.to_csv("../example/result/n94_scoretest_NB_res.tsv", sep="\t", index=False)

# out_pois = map_nominal(
#     dat, family=Poisson(), offset_eta=offset_eta, test=ScoreTest(), score_test=CommonTest(), max_iter=1000
# )

out_nb = map_nominal(
    dat, family=NegativeBinomial(), offset_eta=offset_eta, test=ScoreTest(), mode="nominal", max_iter=600
)

out_lm = map_nominal(dat, family=Gaussian(), offset_eta=0.0, test=ScoreTest(), max_iter=600)

# ~250s
start = timeit.default_timer()
mapcis_out_wald = map_cis(
    dat,
    family=Poisson(),
    offset_eta=offset_eta,
    n_perm=1000,
    robust_se=False,
    compute_qvalue=False,
)
stop = timeit.default_timer()
print("Time: ", stop - start)
mapcis_out_wald.to_csv("./example/result/n94_waldtest_pois_res.tsv", sep="\t", index=False)
