import gzip
import re

from os import PathLike
from typing import Optional, Union

import polars as pl


def validate_user_columns(user_cols, observed_cols) -> list[str]:
    # drop any duplicates and convert back to list
    user_cols = list(set(user_cols))

    # user specifies by column names
    if all(isinstance(c, str) for c in user_cols):
        missing = set(user_cols) - set(observed_cols)
        if missing:
            raise ValueError(f"Columns not found: {sorted(missing)}")
        result = user_cols

    # user specifies by column indices
    elif all(isinstance(c, int) for c in user_cols):
        n = len(observed_cols)
        result = []
        for c in user_cols:
            if c < 0 or c >= n:
                raise ValueError(f"Invalid column index: {c}")
            result.append(observed_cols[c])

    else:
        raise TypeError("user specified columns must be ALL strings (names) or ALL integers (indices)")

    return result


def read_plink_style_tsvlike(
    path_or_filename: Union[str, PathLike],
    keep_columns: Optional[Union[list[str], list[int]]] = None,
    drop_columns: Optional[Union[list[str], list[int]]] = None,
) -> pl.DataFrame:
    """
    Helper function to read in a phenotype or covariate file. Allows for an optional list of column names or column
    indices to be passed in, to parse only a subset of the data.
    """
    iid_re = re.compile(r"^#?iid$", re.IGNORECASE)
    fid_re = re.compile(r"^#?fid$", re.IGNORECASE)

    if keep_columns and drop_columns:
        raise ValueError("Cannot specify both `keep_columns` and `drop_columns`")

    # peek to pull out header, if any
    open_f = gzip.open if str(path_or_filename).endswith(".gz") else open
    with open_f(path_or_filename, "rt") as file:  # type: ignore
        obs_columns = file.readline().split()
        if len(obs_columns) == 0:
            raise ValueError(f"Invalid format for {path_or_filename}")

    # check that IID is present, and drop FID if it is (we never use it)
    iids = [c for c in obs_columns if iid_re.match(c)]
    if len(iids) != 1:
        raise ValueError("Pheno/covar file must contain exactly one IID-like column (e.g., `iid`, `IID`, `#iid`, etc)")

    # if we get here then we have a single match for what the IID-like column is
    iid_col = iids[0]

    # if user provided columns, check validity
    # otherwise use all observed columns
    if keep_columns is not None:
        columns = validate_user_columns(keep_columns, obs_columns)
        user_iids = [c for c in columns if iid_re.match(c)]
        # if iid-like column wasn't provided by user, which we definitely don't expect, add it here for parsing
        # if the user specified it, it should be valid (see above), and we don't need to add it
        if not user_iids:
            columns = [iid_col] + columns
    elif drop_columns is not None:
        # we can use obs_columns bc we know keep_columns wasn't also set.
        columns = validate_user_columns(drop_columns, obs_columns)
        columns = list(set(obs_columns) - set(columns))

        # check that we didnt remove iid column
        iids = [c for c in columns if iid_re.match(c)]
        if not iids:
            columns = [iid_col] + columns
    else:
        columns = obs_columns

    # at this point columns is either all observed columns in data, or appropriate iid colname + specified columns
    # either way lets check for existence of an fid like column and remove it if found, since we dont use it
    columns = [c for c in columns if not fid_re.match(c)]

    # we know at least one column exists due to `obs_columns` and we know that at least iid-like col exists
    # if we've been whittled down to a single column (i.e., iid) then there is no other information and err out
    if len(columns) < 2:
        raise ValueError(f"File {path_or_filename} only has `{iid_col}` column and no other observations")

    df = pl.read_csv(
        path_or_filename,
        separator="\t",
        columns=columns,
        null_values=["NA", "", "NULL", "NaN", "nan"],
    )

    # internally replace iid-like to `iid`
    df = df.rename({iid_col: "iid"})

    return df


def read_offset_tsvlike(
    path_or_filename: Union[str, PathLike],
    column: Optional[Union[str, int]] = None,
) -> pl.DataFrame:
    if column is not None:
        if not isinstance(column, (str, int)):
            raise ValueError(f"Column must be of type `str` or `int`, if not `None`. Found {type(column)}.")
        df_offset = read_plink_style_tsvlike(path_or_filename, keep_columns=[column])  # type: ignore
    else:
        df_offset = read_plink_style_tsvlike(path_or_filename)

    if df_offset.width > 2:
        raise ValueError(f"Offset file {path_or_filename} has multiple columns. Please specify offset column name")
    elif df_offset.width == 1:
        raise ValueError(f"Offset file {path_or_filename} has single column. Please specify valid offset file")

    # we have invariant that width == 2 and iid-like should be in front from `read_plink_style_tsv`
    offname = df_offset.columns[1]
    df_offset = df_offset.rename({offname: "offset"})

    return df_offset
