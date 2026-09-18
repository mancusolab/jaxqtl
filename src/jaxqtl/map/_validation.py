# pattern: Functional Core
"""Validate host-side mapping inputs before fitting or device conversion."""

import numpy as np
import polars as pl


def validate_numeric_frame(frame: pl.DataFrame, label: str) -> None:
    """Require finite numeric observations in every non-IID column."""
    for name, dtype in frame.schema.items():
        if name == "iid":
            continue
        column = frame.get_column(name)
        if not dtype.is_numeric() and dtype != pl.Boolean:
            raise ValueError(f"{label} column {name!r} must be numeric")
        if column.null_count() or not column.cast(pl.Float64).is_finite().all():
            raise ValueError(f"{label} column {name!r} must be finite and nonmissing")


def prepare_covariates(covar: pl.DataFrame, *, one_hot: bool, normalize: bool, intercept: bool) -> pl.DataFrame:
    """Encode and validate the aligned design, then optionally standardize it."""
    if covar.height == 0:
        raise ValueError("No shared samples remain for mapping")
    if any(covar.get_column(name).null_count() for name in covar.columns if name != "iid"):
        raise ValueError("Covariates must be nonmissing in retained samples")
    if one_hot:
        covar = covar.to_dummies(pl.selectors.string().exclude("iid"), drop_first=True)
    validate_numeric_frame(covar, "Covariates")
    names = [name for name in covar.columns if name != "iid"]
    if normalize:
        constant = [name for name in names if covar.get_column(name).n_unique() <= 1]
        if constant:
            raise ValueError(f"Cannot normalize constant covariates: {constant}")
    if intercept:
        if "intercept" in names:
            raise ValueError("Covariates already contain 'intercept'; use --no-intercept or remove that column")
        covar = covar.with_columns(pl.lit(1.0).alias("intercept"))
    if normalize and names:
        cols = pl.col(names).cast(pl.Float64)
        covar = covar.with_columns((cols - cols.mean()) / cols.std())
    validate_numeric_frame(covar, "Covariates")
    values = (
        covar.select(pl.exclude("iid")).to_numpy().astype(float) if covar.width > 1 else np.empty((covar.height, 0))
    )
    n, p = values.shape
    if n <= p + 1:
        raise ValueError(f"Mapping requires more than {p + 1} samples for {p} covariates and a tested variant; got {n}")
    if p:
        # Column scaling prevents different measurement units from dominating the rank tolerance.
        scale = np.max(np.abs(values), axis=0)
        scaled = values / np.where(scale > 0, scale, 1.0)
        if np.linalg.matrix_rank(scaled) < p:
            raise ValueError(
                f"Covariates must have full column rank; remove constant or collinear columns: {covar.columns[1:]}"
            )
    return covar
