# pattern: Functional Core
"""Shared association output fields and numerical validity rules."""

import polars as pl


def association_columns() -> dict:
    """Return common association fields in output order, including optional NB dispersion."""
    return {
        "beta": pl.Float64,
        "se": pl.Float64,
        "pvalue": pl.Float64,
        "nb_alpha": pl.Float64,
        "negloglikelihood": pl.Float64,
        "model_converged": pl.Boolean,
        "result_valid": pl.Boolean,
        "failure_reason": pl.String,
    }


def with_result_diagnostics(frame: pl.DataFrame) -> pl.DataFrame:
    """Annotate numerical validity without dropping rows or conflating it with convergence.

    When several fields fail, the first check below determines the failure reason.
    Cis adjusted p-values are checked when present; convergence flags remain separate.
    """
    checks = [
        (~(pl.col("se").is_finite() & (pl.col("se") > 0)).fill_null(False), "invalid_standard_error"),
        (~pl.col("beta").is_finite().fill_null(False), "nonfinite_effect"),
        (~(pl.col("pvalue").is_finite() & pl.col("pvalue").is_between(0, 1)).fill_null(False), "invalid_pvalue"),
        (~pl.col("negloglikelihood").is_finite().fill_null(False), "nonfinite_objective"),
    ]
    if "nb_alpha" in frame.columns:
        checks.append(
            (~(pl.col("nb_alpha").is_finite() & (pl.col("nb_alpha") >= 0)).fill_null(False), "invalid_dispersion")
        )
    if "pvalue_adj" in frame.columns:
        checks.append(
            (
                ~(pl.col("pvalue_adj").is_finite() & pl.col("pvalue_adj").is_between(0, 1)).fill_null(False),
                "invalid_adjusted_pvalue",
            )
        )
    reason = pl.lit(None, dtype=pl.String)
    for invalid, label in reversed(checks):
        reason = pl.when(invalid).then(pl.lit(label)).otherwise(reason)
    return frame.with_columns(reason.alias("failure_reason")).with_columns(
        pl.col("failure_reason").is_null().alias("result_valid")
    )
