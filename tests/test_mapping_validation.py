# pattern: Functional Core

import numpy as np
import polars as pl
import pytest

import jax

from jaxqtl.map._validation import prepare_covariates
from jaxqtl.map.data import align_on_iid


def test_covariate_normalization_uses_only_shared_samples():
    samples = pl.DataFrame({"iid": ["a", "b", "c", "d"]})
    covar = pl.DataFrame({"iid": ["e", "d", "c", "b", "a"], "age": [1e9, 4.0, 3.0, 2.0, 1.0]})
    covar = align_on_iid([samples, covar])[1]
    result = prepare_covariates(covar, one_hot=False, normalize=True, intercept=True)
    np.testing.assert_allclose(result["age"].to_numpy().mean(), 0, atol=1e-15)
    np.testing.assert_allclose(result["age"].to_numpy().std(ddof=1), 1)
    assert result["iid"].to_list() == ["a", "b", "c", "d"]
    assert result["intercept"].to_list() == [1.0] * 4


def test_single_intercept_is_valid_without_normalization():
    covar = pl.DataFrame({"iid": list("abcd"), "intercept": [1.0] * 4})
    result = prepare_covariates(covar, one_hot=False, normalize=False, intercept=False)
    assert result.equals(covar)


def test_empty_covariate_design_is_valid_without_intercept():
    covar = pl.DataFrame({"iid": list("abcd")})
    result = prepare_covariates(covar, one_hot=False, normalize=False, intercept=False)
    assert result.equals(covar)


def test_categorical_covariates_are_encoded_after_alignment():
    covar = pl.DataFrame({"iid": list("abcde"), "batch": ["a", "b", "a", "b", "a"]})
    result = prepare_covariates(covar, one_hot=True, normalize=True, intercept=True)
    assert result.shape == (5, 3)
    assert "batch" not in result.columns


def test_rank_check_ignores_covariate_measurement_units():
    covar = pl.DataFrame({"iid": list("abcde"), "tiny": [v * 1e-20 for v in [1, 3, 2, 4, 5]]})
    result = prepare_covariates(covar, one_hot=False, normalize=False, intercept=True)
    assert result.height == 5


def test_normalization_rejects_covariates_that_become_collinear_without_intercept():
    covar = pl.DataFrame({"iid": list("abcdef"), "x": range(1, 7), "y": range(2, 8)})
    result = prepare_covariates(covar, one_hot=False, normalize=False, intercept=False)
    assert result.equals(covar)
    with pytest.raises(ValueError, match="full column rank"):
        prepare_covariates(covar, one_hot=False, normalize=True, intercept=False)


@pytest.mark.parametrize("ids", [["a", None], ["a", "a"]])
def test_alignment_rejects_invalid_sample_ids(ids):
    with pytest.raises(ValueError, match="(non-null|Duplicate)"):
        align_on_iid([pl.DataFrame({"iid": ids})])


def test_covariate_validation_rejects_overflow_in_active_precision():
    covar = pl.DataFrame({"iid": list("abcde"), "large": [1e40 * v for v in [1, 3, 2, 4, 5]]})
    with jax.enable_x64(False), pytest.raises(ValueError, match="finite.*precision"):
        prepare_covariates(covar, one_hot=False, normalize=False, intercept=True)


@pytest.mark.parametrize("x64", [False, True])
def test_covariate_rank_validation_in_active_precision(x64):
    covar = pl.DataFrame({"iid": list("abcde"), "x": [1.0, 3.0, 2.0, 4.0, 5.0]})
    with jax.enable_x64(x64):
        result = prepare_covariates(covar, one_hot=False, normalize=False, intercept=True)
        assert result.height == covar.height
        with pytest.raises(ValueError, match="full column rank"):
            prepare_covariates(
                covar.with_columns((pl.col("x") * 2).alias("duplicate")), one_hot=False, normalize=False, intercept=True
            )
