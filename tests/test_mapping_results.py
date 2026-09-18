# pattern: Imperative Shell

import numpy as np
import polars as pl
import pytest

import jax.numpy as jnp

from jaxqtl.hypothesis import ACAT, BetaPermutation, TestResult as AssociationResult
from jaxqtl.map.cis import _empty_cis_columns, _empty_nominal_columns, _process_cis_result, _process_nominal_result
from jaxqtl.map.data import CisData
from jaxqtl.map.trans import _process_result, get_trans_schemas


def _inputs():
    metadata = pl.DataFrame(
        {"chrom": ["1", "1"], "snp": ["rs1", "rs2"], "pos": [101, 102], "a1": ["A", "C"], "a0": ["G", "T"]}
    )
    genotype = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0], [1.0, 1.0]])
    cis = CisData(jnp.ones((4, 1)), genotype, jnp.arange(4.0), jnp.array(0.0), "gene", "1", 100, 103, metadata, 1, 200)
    result = AssociationResult(
        beta=jnp.array([0.1, 0.2]),
        se=jnp.array([0.2, 0.3]),
        p=jnp.array([0.1, 0.2]),
        z=jnp.array([1.0, 2.0]),
        num_iters=jnp.array(2),
        converged=jnp.array(True),
        disp=jnp.array(0.5),
        negloglikelihood=jnp.array(12.0),
    )
    return cis, result


def _trans(cis, result):
    batched = AssociationResult(*(jnp.expand_dims(value, 0) for value in result))
    return _process_result(cis.G, cis.cis_info, batched, [cis.gene_name])[1]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("se", 0.0, "invalid_standard_error"),
        ("se", float("nan"), "invalid_standard_error"),
        ("beta", float("inf"), "nonfinite_effect"),
        ("p", float("nan"), "invalid_pvalue"),
        ("p", 1.1, "invalid_pvalue"),
        ("negloglikelihood", float("inf"), "nonfinite_objective"),
        ("disp", -1.0, "invalid_dispersion"),
    ],
)
def test_diagnostics_agree_across_mapping_modes(field, value, reason):
    cis, result = _inputs()
    changed = jnp.full_like(getattr(result, field), value)
    result = result._replace(**{field: changed})
    nominal = _process_nominal_result(cis, result)
    trans = _trans(cis, result)
    lead = _process_cis_result(cis, result, (jnp.array(0.2), None), 0, gene_test=ACAT())
    for frame in [nominal, trans, pl.DataFrame([lead])]:
        assert frame["result_valid"].to_list() == [False] * frame.height
        assert frame["failure_reason"].to_list() == [reason] * frame.height


def test_nonconvergence_is_separate_from_numerical_validity():
    cis, result = _inputs()
    result = result._replace(converged=jnp.array(False), p=jnp.array([0.0, 1.0]))
    for frame in [_process_nominal_result(cis, result), _trans(cis, result)]:
        assert frame["result_valid"].to_list() == [True, True]
        assert frame["model_converged"].to_list() == [False, False]
        assert frame["failure_reason"].dtype == pl.String
        assert frame["failure_reason"].null_count() == 2


@pytest.mark.parametrize("per_variant", [False, True])
def test_trans_objectives_and_identifiers_follow_phenotype_variant_order(per_variant):
    cis, result = _inputs()
    batched = AssociationResult(*(jnp.stack([value, value]) for value in result))
    objective = jnp.array([[10.0, 11.0], [20.0, 21.0]]) if per_variant else jnp.array([10.0, 20.0])
    batched = batched._replace(negloglikelihood=objective)
    frame = _process_result(cis.G, cis.cis_info, batched, ["g1", "g2"])[1]
    assert frame["phenotype_id"].to_list() == ["g1", "g1", "g2", "g2"]
    np.testing.assert_allclose(frame["negloglikelihood"], [10, 11, 20, 21] if per_variant else [10, 10, 20, 20])
    assert "phenotype" not in frame.columns


def test_common_fields_present_in_all_schemas():
    common = {
        "phenotype_id",
        "snp",
        "beta",
        "se",
        "pvalue",
        "nb_alpha",
        "negloglikelihood",
        "model_converged",
        "result_valid",
        "failure_reason",
    }
    for schema in [_empty_nominal_columns(), _empty_cis_columns(ACAT()), get_trans_schemas()[1]]:
        assert common <= schema.keys()
    permutation = {"shape1", "shape2", "nc_estimate", "perm_converged"}
    assert permutation <= _empty_cis_columns(BetaPermutation()).keys()
    assert not permutation.intersection(_empty_cis_columns(ACAT()))


def test_cis_invalid_adjusted_pvalue_is_diagnosed():
    cis, result = _inputs()
    row = _process_cis_result(cis, result, (jnp.array(float("nan")), None), 0, gene_test=ACAT())
    assert row["result_valid"] is False
    assert row["failure_reason"] == "invalid_adjusted_pvalue"


def test_diagnostic_nulls_round_trip_with_invalid_rows(tmp_path):
    cis, result = _inputs()
    good = _process_nominal_result(cis, result)
    bad = _process_nominal_result(cis, result._replace(se=jnp.array([0.0, float("nan")])))
    combined = pl.concat([good, bad])
    path = tmp_path / "results.parquet"
    combined.write_parquet(path)
    restored = pl.read_parquet(path)
    assert restored["result_valid"].to_list() == [True, True, False, False]
    assert restored["failure_reason"].to_list() == [None, None, "invalid_standard_error", "invalid_standard_error"]
