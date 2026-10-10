# pattern: Functional Core
"""Validate mapping inputs and the covariate design before fitting."""

import polars as pl

import jax
import jax.numpy as jnp

from jax import Array


def _invalid_observations(frame: pl.DataFrame, invalid: pl.Series, message: str) -> None:
    """Report a bounded sample of invalid retained observations."""
    count = int(invalid.sum())
    if count:
        ids = frame.get_column("iid").filter(invalid).head(5).to_list()
        suffix = " ..." if count > 5 else ""
        raise ValueError(f"{message}; {count} retained samples affected; sample IDs: {ids}{suffix}")


def validate_numeric_frame(frame: pl.DataFrame, label: str, *, allow_strings: bool = False) -> None:
    """Require finite numeric observations; optionally accept strings for later encoding."""
    for name, dtype in frame.schema.items():
        if name == "iid":
            continue
        column = frame.get_column(name)
        message = f"{label} column {name!r} must be finite and nonmissing"
        if dtype.is_numeric() or dtype == pl.Boolean:
            _invalid_observations(frame, ~column.cast(pl.Float64).is_finite().fill_null(False), message)
        else:
            _invalid_observations(frame, column.is_null(), message)
            if not (allow_strings and dtype == pl.String):
                raise ValueError(f"{label} column {name!r} must be numeric")


def covariate_array(covar: pl.DataFrame) -> Array:
    """Validate an aligned numeric design and its conversion to active JAX precision."""
    if covar.height == 0:
        raise ValueError("No shared samples remain for mapping")
    if covar.width <= 1:
        raise ValueError("No covariates remain; supply at least one covariate or an intercept column")
    validate_numeric_frame(covar, "Covariates")
    values = covar.select(pl.exclude("iid")).to_jax(
        dtype=pl.Float64 if jax.config.read("jax_enable_x64") else pl.Float32
    )
    finite = jnp.isfinite(values)
    if not bool(finite.all()):
        for index, name in enumerate(name for name in covar.columns if name != "iid"):
            _invalid_observations(
                covar,
                pl.Series((~finite[:, index]).tolist()),
                f"Covariates column {name!r} must remain finite in the active JAX precision",
            )
    return values


def prepare_covariates(covar: pl.DataFrame, *, one_hot: bool, normalize: bool, intercept: bool) -> pl.DataFrame:
    """Encode and optionally standardize the aligned design, then validate it."""
    if covar.height == 0:
        raise ValueError("No shared samples remain for mapping")
    validate_numeric_frame(covar, "Covariates", allow_strings=one_hot)
    if one_hot:
        covar = covar.to_dummies(pl.selectors.string().exclude("iid"), drop_first=True)
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
    n, p = covar.height, covar.width - 1
    if p == 0:
        raise ValueError("No covariates remain; retain at least one covariate or enable the intercept")
    if n <= p + 1:
        raise ValueError(f"Mapping requires more than {p + 1} samples for {p} covariates and a tested variant; got {n}")
    values = covariate_array(covar)
    # Column scaling prevents different measurement units from dominating the rank tolerance.
    scale = jnp.max(jnp.abs(values), axis=0)
    scaled = values / jnp.where(scale > 0, scale, 1.0)
    if jnp.linalg.matrix_rank(scaled) < p:
        names = [name for name in covar.columns if name != "iid"]
        raise ValueError(f"Covariates must have full column rank; remove constant or collinear columns: {names}")
    return covar
