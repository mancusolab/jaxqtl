from jax.config import config

from jaxqtl.families.distribution import Poisson
from jaxqtl.infer.permutation import BetaPerm  # , DirectPerm,
from jaxqtl.io.geno import PlinkReader  # , VCFReader
from jaxqtl.io.pheno import H5AD, PheBedReader, SingleCellFilter
from jaxqtl.io.readfile import read_data
from jaxqtl.map import map_cis

config.update("jax_enable_x64", True)

geno_path = "./example/data/chr22.bed"
raw_count_path = "./example/data/Countdata_n100.h5ad"
covar_path = "./example/data/donor_features.tsv"
pheno_path = "./example/data/CD14_positive_monocyte.bed.gz"
# raw_count_path = "../NextProject/data/OneK1K/Count.h5ad"


# For given cell type, create bed files from h5ad file
pheno_reader = H5AD()
rawcount = pheno_reader(raw_count_path)
count_df = pheno_reader.process(rawcount, SingleCellFilter)

cell_type = "CD14-positive monocyte"
pheno_reader.write_bed(
    count_df,
    gtf_bed_path="./example/data/Homo_sapiens.GRCh37.87.bed.gz",
    out_dir="./example/data",
    cell_type=cell_type,
)


dat = read_data(
    geno_path,
    pheno_path,
    covar_path,
    geno_reader=PlinkReader(),
    pheno_reader=PheBedReader(),
)
dat_CD14 = dat.create_ReadyData()

# TODO: need error handle singlular value (won't stop for now, but Inf estimate in SE)
mapcis_out = map_cis(dat_CD14, family=Poisson(), perm=BetaPerm())
print(mapcis_out.effect_beta)


def test_run_cis_GLM():
    pass
