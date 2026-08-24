import gzip
import re

from os import PathLike

import polars as pl


def validate_user_columns(user_cols, observed_cols) -> list[str]:
    """Validate user-specified column names or indices against observed columns."""
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
    path_or_filename: str | PathLike,
    keep_columns: list[str] | list[int] | None = None,
    drop_columns: list[str] | list[int] | None = None,
) -> pl.DataFrame:
    r"""Read a tab-delimited sample table with a PLINK-style IID column.

    The header must contain exactly one case-insensitive `iid` or `#iid` column.
    An optional `fid` or `#fid` column is dropped, and the IID column is normalized
    to `iid`. The strings `NA`, `NULL`, `NaN`, `nan`, and empty fields are read as
    missing values.

    **Arguments:**

    - `path_or_filename`: Plain-text or gzip-compressed tab-delimited file.
    - `keep_columns`: Optional column names or zero-based indices to retain. The IID
      column is added automatically when omitted.
    - `drop_columns`: Optional column names or zero-based indices to remove. The IID
      column is retained even when requested for removal.

    **Returns:**

    A Polars frame with normalized `iid` and at least one data column.

    **Raises:**

    - `ValueError`: If keep and drop selections are both supplied, the header is
      empty, exactly one IID-like column is not present, a requested column is
      missing, or no data columns remain.
    - `TypeError`: If a column selection mixes names and indices.
    """
    iid_re = re.compile(r"^#?iid$", re.IGNORECASE)
    fid_re = re.compile(r"^#?fid$", re.IGNORECASE)

    if keep_columns and drop_columns:
        raise ValueError("Cannot specify both `keep_columns` and `drop_columns`")

    # peek to pull out header, if any
    name = str(path_or_filename)
    open_f = gzip.open if name.endswith(".gz") else open
    with open_f(name, "rt") as file:
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
        name,
        separator="\t",
        columns=columns,
        null_values=["NA", "", "NULL", "NaN", "nan"],
    )

    # internally replace iid-like to `iid`
    df = df.rename({iid_col: "iid"})

    return df


def read_offset_tsvlike(
    path_or_filename: str | PathLike,
    column: str | int | None = None,
) -> pl.DataFrame:
    r"""Read a sample-aligned offset from a tab-delimited file.

    **Arguments:**

    - `path_or_filename`: Plain-text or gzip-compressed table accepted by
      `read_plink_style_tsvlike`.
    - `column`: Name or zero-based index of the offset column. When omitted, the
      input must already contain exactly one non-IID column.

    **Returns:**

    A two-column Polars frame with columns `iid` and `offset`.

    **Raises:**

    - `ValueError`: If `column` has an unsupported type or the selected table does
      not contain exactly one non-IID column.
    - `TypeError`: If column selection is invalid for
      `read_plink_style_tsvlike`.
    """
    if column is not None:
        if not isinstance(column, str | int):
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


def read_single_column_file(
    path_or_filename: str | PathLike,
) -> list:
    r"""Read stripped lines from a plain-text or gzip-compressed file.

    A first line beginning with `#` is treated as a header and skipped. Later lines,
    including empty or comment-like lines, are returned after stripping whitespace.

    **Arguments:**

    - `path_or_filename`: Input text file, optionally ending in `.gz`.

    **Returns:**

    A list of strings in file order.
    """
    output = []
    name = str(path_or_filename)
    open_f = gzip.open if name.endswith(".gz") else open

    with open_f(name, "rt") as file:
        first = file.readline()
        if first and first[0] != "#":
            output.append(first.strip())
        for line in file:
            output.append(line.strip())

    return output
