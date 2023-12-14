from typing import Optional

import pandas as pd


def covar_reader(
    covar_path: str, add_covar_path: Optional[str] = None, covar_test: Optional[str] = None
) -> pd.DataFrame:
    """Read covariate file
    default is long format:
    id UNR1 UNR2 UNR3 UNR4

    wide format (tensorqtl input):
    id . .
    varname1 . .
    varname2 . .

    Note: no missing values allowed

    :param covar_path: covariate path, allow bed format and tsv format
    :param add_covar_path: path for additional covariates to add, allow only tsv format
    :param covar_test: covariate to test for association against gene expression
    :return: data frame of covariates
    """
    if covar_path.endswith((".bed.gz", ".bed")):
        # wide format
        covar = pd.read_csv(covar_path, sep="\t", index_col=0).T
        covar.columns.name = None  # remove redundant name due to transpose
        covar.index.names = ["iid"]
    elif covar_path.endswith((".tsv", ".tsv.gz")):
        # long format
        covar = pd.read_csv(covar_path, sep="\t", index_col=0)
        covar.index.names = ["iid"]
    else:
        raise ValueError("Unsupported covariate file type.")

    if add_covar_path is not None:
        add_covar = pd.read_csv(add_covar_path, sep="\t", index_col=0)
        add_covar.index.names = ["iid"]
        covar = covar.join(add_covar, how="left")  # join on index

    if covar_test is not None:
        # put covar_test in the last column
        col_names = covar.columns.to_list()
        col_names.remove(covar_test)
        col_names.append(covar_test)
        covar = covar[col_names]

    assert not covar.isnull().values.any(), "Missing values are not allowed in covariate file."

    return covar
