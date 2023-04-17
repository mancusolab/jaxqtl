# import numpy.testing as nptest
#
# import numpy as np
# from statsmodels.discrete.discrete_model import (  # , NegativeBinomial as smNB
#     Poisson as smPoisson,
# )
#
# import jax.numpy as jnp
# from jax import random as rdm
import os

import pandas as pd

from jax.config import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.infer.permutation import BetaPerm
from jaxqtl.io.geno import PlinkReader
from jaxqtl.io.pheno import H5AD, PheBedReader, SingleCellFilter
from jaxqtl.io.readfile import read_data
from jaxqtl.map import map_cis, map_cis_nominal

config.update("jax_enable_x64", True)
# pd.set_option("display.max_columns", 500)

geno_path = "./example/data/chr22.n94.bed"
raw_count_path = "./example/local/Countdata_n100.h5ad"
covar_path = "./example/data/donor_features.n94.tsv"
pheno_path = "./example/data/CD14_positive_monocyte.n94.bed.gz"
# raw_count_path = "../NextProject/data/OneK1K/Count.h5ad"


# Prepare input #
# For given cell type, create bed files from h5ad file
pheno_reader = H5AD()
count_mat = pheno_reader(raw_count_path)
count_df = pheno_reader.process(count_mat, SingleCellFilter)

# cell_type = "CD14-positive monocyte"
pheno_reader.write_bed(
    count_df,
    gtf_bed_path="./example/data/Homo_sapiens.GRCh37.87.bed.gz",
    out_dir="./example/local/phe_bed",
    celltype_path="./example/data/celltype.tsv",
)


# run Mapping #
dat = read_data(
    geno_path,
    pheno_path,
    covar_path,
    geno_reader=PlinkReader(),
    pheno_reader=PheBedReader(),
)

dat.filter_geno("22")

# TODO: need error handle singular value (won't stop for now, but Inf estimate in SE)
mapcis_out = map_cis(dat, family=Poisson(), perm=BetaPerm())
print(mapcis_out.slope)

map_cis_nominal(dat, family=Poisson(), out_path="./example/result/dat_n94")

prefix = "dat_n94"
out_dir = "./example/result"
pairs_df = pd.read_parquet(os.path.join(out_dir, f"{prefix}.cis_qtl_pairs.22.parquet"))

# mapcis_df = prepare_cis_output(dat, mapcis_out)
# print(mapcis_df)


# def cis_scan_sm(X, G, y):
#     """
#     run GLM across variants in a flanking window of given gene
#     cis-widow: plus and minus W base pairs, total length 2*cis_window
#     """
#     beta = []
#     se = []
#     p = []
#
#     for snp in np.array(G.T):
#         M = np.hstack((np.array(X), snp[:, np.newaxis]))
#         glmstate = smPoisson(y, M).fit(disp=0)
#         beta.append(glmstate.params[-1])
#         se.append(glmstate.bse[-1])
#         p.append(glmstate.pvalues[-1])
#
#     return CisGLMState(
#         beta=jnp.asarray(beta),
#         se=jnp.asarray(se),
#         p=jnp.asarray(p),
#         num_iters=jnp.array([-9]),
#         converged=jnp.array([-9]),
#     )
#
#
# def map_cis_nominal_sm(
#     dat,
#     seed: int = 123,
#     window: int = 500000,
# ):
#     n, k = dat.covar.shape
#     gene_info = dat.pheno_meta
#
#     # append genotype as the last column
#     X = jnp.hstack((jnp.ones((n, 1)), dat.covar))
#     key = rdm.PRNGKey(seed)
#
#     effect_beta = []
#     beta_se = []
#     nominal_p = []
#
#     for gene in gene_info:
#         gene_name, chrom, start_min, end_max = gene
#         lstart = min(0, start_min - window)
#         rend = end_max + window
#
#         # pull cis G and y for this gene
#         G, y = _setup_G_y(dat, gene_name, str(chrom), lstart, rend)
#
#         # skip if no cis SNPs found
#         if G.shape[1] == 0:
#             continue
#
#         key, g_key = rdm.split(key)
#
#         result = cis_scan_sm(np.array(X), np.array(G), np.array(y))
#
#         # combine results
#         effect_beta.append(result.beta)
#         beta_se.append(result.se)
#         nominal_p.append(result.p)
#
#         # unit test for 4 genes
#         if len(nominal_p) > 3:
#             break
#
#     return MapCis_OutState(
#         effect_beta=effect_beta,
#         beta_se=beta_se,
#         nominal_p=nominal_p,
#         adj_p=[],
#         beta_param=[],
#         converged=[],
#     )


# %timeit -n1 -r1 mapcis_out_jaxqtl = map_cis_nominal(dat_CD14, family=Poisson())
# %timeit -n1 -r1 mapcis_out_sm = map_cis_nominal_sm(dat_CD14)


# def test_run_cis_GLM():
#     mapcis_out_jaxqtl = map_cis_nominal(dat_CD14, family=Poisson())
#     mapcis_out_sm = map_cis_nominal_sm(dat_CD14)
#     nptest.assert_allclose(mapcis_out_jaxqtl.effect_beta, mapcis_out_sm.effect_beta, rtol=1e-5)
