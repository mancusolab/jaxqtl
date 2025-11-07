from typing import Optional

import pandas as pd


def covar_reader(
    covar_path: str,
    covar_test: Optional[str] = None,
    rm_covar: Optional[str] = None,
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
    :param covar_test: covariate to test for association against gene expression
    :return: data frame of covariates
    """
    if covar_path.endswith((".bed.gz", ".bed", ".txt")):
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

    if covar_test is not None:
        covar = covar[covar_test]

    # remove covariates (especially for sex-specific analysis, drop the sex variable)
    if rm_covar is not None:
        covar = covar.drop(rm_covar, axis=1)

    # todo: should we allow imputation?
    if covar.isnull().values.any():
        raise ValueError("Missing values are not allowed in covariate file.")

    return covar
