import pandas as pd


def covar_reader(covar_path: str) -> pd.DataFrame:
    """Read covariate file
    default is long format:
    id UNR1 UNR2 UNR3 UNR4

    wide format (tensorqtl input):
    id . .
    varname1 . .
    varname2 . .

    Note: no missing values allowed
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

    assert not covar.isnull().values.any(), "Missing values are not allowed in covariate file."

    return covar
