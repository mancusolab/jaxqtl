import pandas as pd
from sklearn.decomposition import PCA

from jaxqtl.io.pheno import bed_transform_y

# from numpy import random

cell_meta = pd.read_csv(
    "/project/nmancuso_8/elezhang/projects/jaxqtl/data/pheno_meta/celltype_14.tsv",
    header=None,
)

cell_meta = cell_meta.iloc[:, 0].values

cell_meta = [x.replace(" ", "_") for x in cell_meta]

method = "tmm"
pheno_path = "./data/pheno/celltype16/"
covar_path = "./data/features/"

for celltype in cell_meta:
    # celltype = "NK"
    bed_tmm = bed_transform_y(pheno_path + f"{celltype}.bed.gz", method=method)

    bed_tmm.to_csv(pheno_path + f"{celltype}.{method}.bed.gz", sep="\t", index=False)

    # calculate PC and create bed file for tensorqtl
    count_std = bed_tmm.iloc[:, 4:].T  # nxM
    count_std = (count_std - count_std.mean()) / count_std.std()  # standardize genes

    pca_pheno = PCA(n_components=2)
    pca_pheno.fit(count_std)
    PCs = pca_pheno.fit_transform(count_std)  # nxk

    covar = pd.read_csv(covar_path + "donor_features.all.6PC.tsv", sep="\t")

    PCs_df = pd.DataFrame(PCs)
    PCs_df.columns = ["E_PC1", "E_PC2"]
    PCs_df["iid"] = count_std.index

    covar = covar.merge(PCs_df, how="right", on="iid")

    covar = covar.T  # flip to put individuals on columns
    covar.to_csv(
        covar_path + f"donor_features.all.6PC.{celltype}.PC.bed",
        sep="\t",
        index=True,
        header=False,
    )

    print(celltype)

# # for each gene permute the columns
# bed_tmm_perm = bed_tmm.iloc[:, 4:]  # nxM
#
# random.seed(1)
# for i in range(len(bed_tmm_perm)):
#     bed_tmm_perm.iloc[i] = random.permutation(bed_tmm_perm.iloc[i])
#     print(i)
#
# bed_tmm.iloc[:, 4:] = bed_tmm_perm
# bed_tmm.to_csv(pheno_path+f"{celltype}.{method}.perm.seed1.bed.gz", sep='\t', index=False)
