# pattern: Functional Core

import math

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import numpy as np
import pytest

from scipy import sparse

import jaxqtl


MAX_EXACT_FLOAT64_INT = 2**53


class _DenseConversionForbiddenCSR(sparse.csr_array):
    def toarray(self, *args, **kwargs):
        raise AssertionError("operator construction must not densify counts")

    def todense(self, *args, **kwargs):
        raise AssertionError("operator construction must not densify counts")


def _state_api():
    assert hasattr(jaxqtl, "state"), "jaxqtl.state is not available from the root package"
    return jaxqtl.state


def _counts(values: list[list[int]], *, dtype=np.int64) -> sparse.csr_array:
    return sparse.csr_array(np.asarray(values, dtype=dtype))


def _reference_alpha(
    counts: sparse.csr_array,
    chromosomes: np.ndarray,
    excluded_chromosome: str | None,
) -> tuple[float, int, int]:
    dense = np.asarray(counts.toarray(), dtype=np.float64)
    keep = np.ones(dense.shape[1], dtype=np.bool_)
    if excluded_chromosome is not None:
        keep = chromosomes != excluded_chromosome
    retained = dense[:, keep]
    means = np.mean(retained, axis=0)
    variances = np.var(retained, axis=0, ddof=1)
    valid = np.isfinite(means) & np.isfinite(variances) & (means > 0.0) & (variances > 0.0)
    numerator = math.fsum(
        float(mean * mean * (variance - mean)) for mean, variance in zip(means[valid], variances[valid])
    )
    denominator = math.fsum(float(mean**4) for mean in means[valid])
    excluded = 0
    if excluded_chromosome is not None:
        all_means = np.mean(dense, axis=0)
        all_variances = np.var(dense, axis=0, ddof=1)
        all_valid = np.isfinite(all_means) & np.isfinite(all_variances) & (all_means > 0.0) & (all_variances > 0.0)
        excluded = int(np.count_nonzero(all_valid & (chromosomes == excluded_chromosome)))
    return numerator / denominator, int(np.count_nonzero(valid)), excluded


def _positive_fixture() -> tuple[sparse.csr_array, np.ndarray]:
    counts = _counts(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 2, 0, 0],
            [4, 8, 2, 0, 2],
        ]
    )
    chromosomes = np.asarray(["1", "2", "3", "4", "X"])
    return counts, chromosomes


def _operator_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray(
        [
            [0, 1, 0, 2],
            [3, 0, 1, 0],
            [1, 2, 0, 0],
            [0, 0, 4, 1],
            [2, 1, 0, 3],
        ],
        dtype=np.int64,
    )
    chromosomes = np.asarray(["1", "2", "1", "X"])
    donor_index = np.asarray([1, 0, 1, 0, 2], dtype=np.int64)
    return counts, chromosomes, donor_index


def _dense_operator_reference(
    counts: np.ndarray,
    chromosomes: np.ndarray,
    donor_index: np.ndarray,
    *,
    alpha: float,
    excluded_chromosome: str | None = None,
    center_donors: bool = True,
    balance_donors: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.ones(counts.shape[1], dtype=np.bool_)
    if excluded_chromosome is not None:
        active = chromosomes != excluded_chromosome
    selected = counts[:, active]
    transformed = np.log1p(4.0 * alpha * selected.astype(np.float64))
    n_cells, n_genes = transformed.shape
    clr = np.eye(n_genes) - np.ones((n_genes, n_genes)) / n_genes
    operator = transformed @ clr

    n_donors = int(np.max(donor_index)) + 1
    donor_counts = np.bincount(donor_index, minlength=n_donors)
    if center_donors:
        centered = np.empty_like(operator)
        for donor in range(n_donors):
            rows = donor_index == donor
            centered[rows] = operator[rows] - np.mean(operator[rows], axis=0)
        operator = centered

    weights = np.ones(n_cells, dtype=np.float64)
    if balance_donors:
        weights = np.zeros(n_cells, dtype=np.float64)
        for donor, count in enumerate(donor_counts):
            if count >= 2:
                weights[donor_index == donor] = 1.0 / (n_donors * (count - 1))
        operator = np.sqrt(weights)[:, None] * operator
    return operator, np.flatnonzero(active), weights


def test_pflog_statistics_api_is_available_from_state_package() -> None:
    state = _state_api()

    assert hasattr(state, "compute_pflog_statistics")
    assert hasattr(state, "estimate_pflog_alpha")
    assert {"compute_pflog_statistics", "estimate_pflog_alpha"} <= set(state.__all__)


def test_statistics_match_hand_derived_means_variances_and_terms() -> None:
    state = _state_api()
    counts = _counts(
        [
            [0, 1, 2, 0],
            [0, 1, 2, 0],
            [3, 2, 2, 0],
        ]
    )

    statistics = state.compute_pflog_statistics(counts, np.asarray(["1", "2", "3", "4"]))

    expected_means = np.asarray([1.0, 4.0 / 3.0, 2.0, 0.0])
    expected_variances = np.asarray([3.0, 1.0 / 3.0, 0.0, 0.0])
    np.testing.assert_allclose(statistics.means, expected_means, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(statistics.variances, expected_variances, rtol=0.0, atol=2e-16)
    np.testing.assert_array_equal(statistics.retained_gene_mask, [True, True, False, False])
    np.testing.assert_allclose(
        statistics.a_terms,
        expected_means**2 * (expected_variances - expected_means),
        rtol=1e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(statistics.b_terms, expected_means**4, rtol=1e-15, atol=0.0)
    assert statistics.total_numerator == pytest.approx(math.fsum(statistics.a_terms[:2]))
    assert statistics.total_denominator == pytest.approx(math.fsum(statistics.b_terms[:2]))


def test_statistics_preserve_unit_scale_variance_at_exact_float64_boundary() -> None:
    state = _state_api()
    counts = _counts(
        [[MAX_EXACT_FLOAT64_INT - 1], [MAX_EXACT_FLOAT64_INT]],
        dtype=np.uint64,
    )

    statistics = state.compute_pflog_statistics(counts, np.asarray(["1"]))

    assert statistics.means[0] == float(MAX_EXACT_FLOAT64_INT)
    assert statistics.mean_error_scales[0] >= 0.5
    assert statistics.variances[0] == 0.5
    expected_tolerance = 64.0 * np.finfo(np.float64).eps * max(0.5, 1.0)
    assert statistics.variance_roundoff_tolerances[0] == pytest.approx(expected_tolerance)
    assert statistics.retained_gene_mask[0]


def test_statistics_include_implicit_zeros_and_exclude_exact_constants() -> None:
    state = _state_api()
    counts = _counts(
        [
            [0, 7, 0],
            [0, 7, 0],
            [3, 7, 0],
        ]
    )

    statistics = state.compute_pflog_statistics(counts, np.asarray(["1", "2", "3"]))

    np.testing.assert_allclose(statistics.means, [1.0, 7.0, 0.0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(statistics.variances, [3.0, 0.0, 0.0], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(statistics.retained_gene_mask, [True, False, False])


def test_statistics_cache_compensated_autosome_and_unassigned_bins() -> None:
    state = _state_api()
    counts, chromosomes = _positive_fixture()

    statistics = state.compute_pflog_statistics(counts, chromosomes)

    for chromosome in range(1, 23):
        selected = statistics.retained_gene_mask & (chromosomes == str(chromosome))
        assert statistics.autosome_numerators[chromosome - 1] == math.fsum(statistics.a_terms[selected])
        assert statistics.autosome_denominators[chromosome - 1] == math.fsum(statistics.b_terms[selected])
        assert statistics.autosome_gene_counts[chromosome - 1] == np.count_nonzero(selected)
    unassigned = statistics.retained_gene_mask & ~np.isin(chromosomes, [str(value) for value in range(1, 23)])
    assert statistics.unassigned_numerator == math.fsum(statistics.a_terms[unassigned])
    assert statistics.unassigned_denominator == math.fsum(statistics.b_terms[unassigned])
    assert statistics.unassigned_gene_count == np.count_nonzero(unassigned)


def test_statistics_results_are_immutable_including_cached_arrays() -> None:
    state = _state_api()
    counts, chromosomes = _positive_fixture()
    statistics = state.compute_pflog_statistics(counts, chromosomes)

    with pytest.raises(FrozenInstanceError):
        statistics.n_cells = 99
    with pytest.raises(ValueError, match="read-only"):
        statistics.means[0] = 0.0


def test_fast_loco_matches_physical_chromosome_removal_for_every_autosome() -> None:
    state = _state_api()
    counts, chromosomes = _positive_fixture()
    statistics = state.compute_pflog_statistics(counts, chromosomes)

    for chromosome in (str(value) for value in range(1, 23)):
        reference_alpha, retained_count, excluded_count = _reference_alpha(counts, chromosomes, chromosome)

        diagnostics = state.estimate_pflog_alpha(statistics, excluded_chromosome=chromosome)

        assert diagnostics.alpha == pytest.approx(reference_alpha, rel=2e-15)
        assert diagnostics.retained_gene_count == retained_count
        assert diagnostics.excluded_gene_count == excluded_count
        assert diagnostics.numerator == pytest.approx(
            diagnostics.difference_numerator,
            abs=diagnostics.numerator_tolerance,
        )
        assert diagnostics.denominator == pytest.approx(
            diagnostics.difference_denominator,
            abs=diagnostics.denominator_tolerance,
        )


def test_fixed_cipher_through_origin_reference_fixture() -> None:
    state = _state_api()
    counts = _counts(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [4, 8],
        ]
    )
    statistics = state.compute_pflog_statistics(counts, np.asarray(["1", "2"]))

    diagnostics = state.estimate_pflog_alpha(statistics)

    assert diagnostics.alpha == pytest.approx(59.0 / 17.0)
    assert diagnostics.numerator == pytest.approx(59.0)
    assert diagnostics.denominator == pytest.approx(17.0)
    assert diagnostics.source == "auto"
    assert diagnostics.excluded_chromosome is None


def test_sub_poisson_genes_remain_in_fit_with_negative_numerator_terms() -> None:
    state = _state_api()
    counts, chromosomes = _positive_fixture()

    statistics = state.compute_pflog_statistics(counts, chromosomes)
    diagnostics = state.estimate_pflog_alpha(statistics)

    assert statistics.retained_gene_mask[2]
    assert statistics.a_terms[2] < 0.0
    assert diagnostics.retained_gene_count == 4
    assert diagnostics.alpha > 0.0


def test_absent_canonical_autosome_is_a_noop_exclusion() -> None:
    state = _state_api()
    counts, chromosomes = _positive_fixture()
    statistics = state.compute_pflog_statistics(counts, chromosomes)

    genomewide = state.estimate_pflog_alpha(statistics)
    absent = state.estimate_pflog_alpha(statistics, excluded_chromosome="22")

    assert absent.alpha == genomewide.alpha
    assert absent.numerator == genomewide.numerator
    assert absent.denominator == genomewide.denominator
    assert absent.excluded_gene_count == 0
    assert absent.excluded_numerator == 0.0
    assert absent.excluded_denominator == 0.0


@pytest.mark.parametrize("excluded_chromosome", ["chr1", "01", "X", "MT", 1])
def test_rejects_noncanonical_loco_chromosomes(excluded_chromosome) -> None:
    state = _state_api()
    counts, chromosomes = _positive_fixture()
    statistics = state.compute_pflog_statistics(counts, chromosomes)

    with pytest.raises(ValueError, match="canonical autosome"):
        state.estimate_pflog_alpha(statistics, excluded_chromosome=excluded_chromosome)


def test_rejects_too_few_cells() -> None:
    state = _state_api()

    with pytest.raises(ValueError, match="at least two cells"):
        state.compute_pflog_statistics(_counts([[3]]), np.asarray(["1"]))


def test_rejects_nonpositive_automatic_numerator_without_clipping() -> None:
    state = _state_api()
    statistics = state.compute_pflog_statistics(_counts([[1], [2]]), np.asarray(["1"]))

    with pytest.raises(ValueError, match="numerator.*strictly positive"):
        state.estimate_pflog_alpha(statistics)


def test_rejects_degenerate_automatic_denominator_without_fallback() -> None:
    state = _state_api()
    statistics = state.compute_pflog_statistics(_counts([[2], [2], [2]]), np.asarray(["1"]))

    with pytest.raises(ValueError, match="denominator.*strictly positive"):
        state.estimate_pflog_alpha(statistics)


@pytest.mark.parametrize("override", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_or_nonpositive_overrides(override: float) -> None:
    state = _state_api()
    counts, chromosomes = _positive_fixture()
    statistics = state.compute_pflog_statistics(counts, chromosomes)

    with pytest.raises(ValueError, match="override.*finite and strictly positive"):
        state.estimate_pflog_alpha(statistics, override=override)


def test_valid_override_bypasses_invalid_automatic_fit_and_retains_diagnostics() -> None:
    state = _state_api()
    statistics = state.compute_pflog_statistics(_counts([[1], [2]]), np.asarray(["1"]))

    diagnostics = state.estimate_pflog_alpha(statistics, excluded_chromosome="2", override=0.125)

    assert diagnostics.alpha == 0.125
    assert diagnostics.source == "override"
    assert diagnostics.excluded_chromosome == "2"
    assert diagnostics.retained_gene_count == 1
    assert diagnostics.excluded_gene_count == 0
    assert diagnostics.numerator < 0.0
    assert diagnostics.denominator > 0.0


def test_pflog_operator_api_is_available_from_state_package() -> None:
    state = _state_api()

    assert hasattr(state, "PFLogOperator")
    assert hasattr(state, "pflog_operator")
    assert state.__all__ == [
        "PFLogOperator",
        "compute_pflog_statistics",
        "estimate_pflog_alpha",
        "pflog_operator",
    ]


def test_operator_vector_and_block_actions_match_dense_reference() -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    dense, active_indices, _ = _dense_operator_reference(
        counts,
        chromosomes,
        donor_index,
        alpha=0.75,
        excluded_chromosome="1",
    )
    operator = state.pflog_operator(
        sparse.csr_array(counts),
        chromosomes,
        donor_index,
        alpha=0.75,
        excluded_chromosome="1",
    )
    vector = np.asarray([1.25, -0.5])
    cell_vector = np.asarray([0.5, -1.0, 2.0, 0.25, -0.75])
    block = np.asarray([[1.0, 0.0, -0.25], [0.5, -2.0, 1.5]])
    cell_block = np.arange(15, dtype=np.float64).reshape(5, 3) / 7.0

    assert isinstance(operator, state.PFLogOperator)
    assert operator.shape == dense.shape
    assert operator.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(operator.diagnostics.active_gene_indices, active_indices)
    np.testing.assert_allclose(operator.matvec(vector), dense @ vector, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(operator.rmatvec(cell_vector), dense.T @ cell_vector, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(operator.matmat(block), dense @ block, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(operator.rmatmat(cell_block), dense.T @ cell_block, rtol=2e-15, atol=2e-15)


def test_operator_satisfies_inner_product_adjoint_identity() -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    operator = state.pflog_operator(
        sparse.csr_array(counts),
        chromosomes,
        donor_index,
        alpha=0.25,
    )
    feature_vector = np.asarray([0.5, -1.0, 0.25, 2.0])
    cell_vector = np.asarray([-0.25, 1.0, 1.5, -2.0, 0.75])

    forward_inner_product = np.vdot(operator.matvec(feature_vector), cell_vector)
    adjoint_inner_product = np.vdot(feature_vector, operator.rmatvec(cell_vector))

    assert forward_inner_product == pytest.approx(adjoint_inner_product, rel=2e-15, abs=2e-15)


def test_excluded_chromosome_cannot_affect_values_or_clr_center() -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    perturbed = counts.copy()
    perturbed[:, chromosomes == "1"] = np.asarray([[10**8, 10**7]])

    baseline = state.pflog_operator(
        sparse.csr_array(counts),
        chromosomes,
        donor_index,
        alpha=0.5,
        excluded_chromosome="1",
        center_donors=False,
        balance_donors=False,
    )
    changed = state.pflog_operator(
        sparse.csr_array(perturbed),
        chromosomes,
        donor_index,
        alpha=0.5,
        excluded_chromosome="1",
        center_donors=False,
        balance_donors=False,
    )
    active_basis = np.eye(2)

    np.testing.assert_array_equal(baseline.diagnostics.active_gene_indices, [1, 3])
    np.testing.assert_array_equal(changed.diagnostics.active_gene_indices, [1, 3])
    np.testing.assert_allclose(baseline.matmat(active_basis), changed.matmat(active_basis), rtol=0.0, atol=0.0)


def test_donor_centered_features_have_zero_means_for_nonsorted_membership() -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    operator = state.pflog_operator(
        sparse.csr_array(counts),
        chromosomes,
        donor_index,
        alpha=1.0,
        balance_donors=False,
    )
    centered_features = operator.matmat(np.eye(counts.shape[1]))

    for donor in range(operator.diagnostics.n_donors):
        np.testing.assert_allclose(
            np.mean(centered_features[donor_index == donor], axis=0),
            np.zeros(counts.shape[1]),
            rtol=0.0,
            atol=2e-16,
        )


def test_balanced_covariance_matches_explicit_donor_formula_and_zeros_singletons() -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    operator = state.pflog_operator(
        sparse.csr_array(counts),
        chromosomes,
        donor_index,
        alpha=0.5,
    )
    centered, _, _ = _dense_operator_reference(
        counts,
        chromosomes,
        donor_index,
        alpha=0.5,
        balance_donors=False,
    )
    n_donors = operator.diagnostics.n_donors
    expected_covariance = np.zeros((counts.shape[1], counts.shape[1]), dtype=np.float64)
    for donor in range(n_donors):
        donor_values = centered[donor_index == donor]
        if donor_values.shape[0] >= 2:
            expected_covariance += donor_values.T @ donor_values / (n_donors * (donor_values.shape[0] - 1))
    balanced_features = operator.matmat(np.eye(counts.shape[1]))

    np.testing.assert_allclose(balanced_features.T @ balanced_features, expected_covariance, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(balanced_features[donor_index == 2], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(operator.diagnostics.cell_weights[donor_index == 2], 0.0, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("center_donors", [True, False])
def test_unbalanced_centering_toggles_match_dense_reference(center_donors: bool) -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    dense, _, expected_weights = _dense_operator_reference(
        counts,
        chromosomes,
        donor_index,
        alpha=0.5,
        center_donors=center_donors,
        balance_donors=False,
    )
    operator = state.pflog_operator(
        sparse.csr_array(counts),
        chromosomes,
        donor_index,
        alpha=0.5,
        center_donors=center_donors,
        balance_donors=False,
    )

    np.testing.assert_allclose(operator.matmat(np.eye(counts.shape[1])), dense, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(operator.diagnostics.cell_weights, expected_weights)
    assert operator.config.center_donors is center_donors
    assert not operator.config.balance_donors


def test_balancing_zeros_singletons_even_when_donor_centering_is_disabled() -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    operator = state.pflog_operator(
        sparse.csr_array(counts),
        chromosomes,
        donor_index,
        alpha=0.5,
        center_donors=False,
    )

    transformed = operator.matmat(np.eye(counts.shape[1]))

    np.testing.assert_allclose(transformed[donor_index == 2], 0.0, rtol=0.0, atol=0.0)


def test_operator_sparse_storage_scales_with_input_nnz_without_dense_conversion() -> None:
    state = _state_api()
    dense_counts, chromosomes, donor_index = _operator_fixture()
    counts = _DenseConversionForbiddenCSR(sparse.csr_array(dense_counts))

    operator = state.pflog_operator(counts, chromosomes, donor_index, alpha=0.5, excluded_chromosome="1")

    expected_nnz = sparse.csr_array(dense_counts[:, chromosomes != "1"]).nnz
    assert operator.diagnostics.input_nnz == counts.nnz
    assert operator.diagnostics.transformed_nnz == expected_nnz
    assert operator.diagnostics.transformed_nnz <= operator.diagnostics.input_nnz
    assert operator.diagnostics.transformed_shape == (counts.shape[0], 2)


def test_extreme_finite_alpha_count_transform_matches_high_precision_reference() -> None:
    state = _state_api()
    alpha = float(np.finfo(np.float64).max)
    count = MAX_EXACT_FLOAT64_INT
    counts = _counts([[count, 1], [0, 0]], dtype=np.uint64)
    operator = state.pflog_operator(
        counts,
        np.asarray(["1", "2"]),
        np.asarray([0, 0]),
        alpha=alpha,
        center_donors=False,
        balance_donors=False,
    )
    with localcontext() as context:
        context.prec = 100
        expected_large = float((Decimal(1) + Decimal(4) * Decimal.from_float(alpha) * Decimal(count)).ln())
        expected_small = float((Decimal(1) + Decimal(4) * Decimal.from_float(alpha)).ln())

    result = operator.matvec(np.asarray([1.0, -1.0]))

    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(expected_large - expected_small, rel=2e-15, abs=2e-15)
    assert result[1] == 0.0


@pytest.mark.parametrize("method", ["matvec", "matmat"])
def test_feature_centering_is_overflow_safe_for_equal_maximum_inputs(method: str) -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    operator = state.pflog_operator(
        sparse.csr_array(counts),
        chromosomes,
        donor_index,
        alpha=0.5,
        center_donors=False,
        balance_donors=False,
    )
    maximum = np.finfo(np.float64).max
    values = np.full(operator.shape[1], maximum)
    if method == "matmat":
        values = np.column_stack((values, values))

    result = getattr(operator, method)(values)

    np.testing.assert_array_equal(result, np.zeros_like(result))


@pytest.mark.parametrize("method", ["rmatvec", "rmatmat"])
def test_donor_centering_is_overflow_safe_for_equal_maximum_inputs(method: str) -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    operator = state.pflog_operator(
        sparse.csr_array(counts),
        chromosomes,
        donor_index,
        alpha=0.5,
        balance_donors=False,
    )
    maximum = np.finfo(np.float64).max
    values = np.full(operator.shape[0], maximum)
    if method == "rmatmat":
        values = np.column_stack((values, values))

    result = getattr(operator, method)(values)

    np.testing.assert_array_equal(result, np.zeros_like(result))


@pytest.mark.parametrize(
    ("method", "shape"),
    [
        ("matvec", (4,)),
        ("rmatvec", (5,)),
        ("matmat", (4, 2)),
        ("rmatmat", (5, 2)),
    ],
)
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_operator_actions_reject_nonfinite_inputs(method: str, shape: tuple[int, ...], nonfinite: float) -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    operator = state.pflog_operator(sparse.csr_array(counts), chromosomes, donor_index, alpha=0.5)
    values = np.zeros(shape, dtype=np.float64)
    values.flat[0] = nonfinite

    with pytest.raises(ValueError, match=rf"{method} input.*finite"):
        getattr(operator, method)(values)


def test_operator_actions_reject_nonfinite_results_from_finite_inputs() -> None:
    state = _state_api()
    counts = _counts([[1, 0], [0, 0]])
    operator = state.pflog_operator(
        counts,
        np.asarray(["1", "2"]),
        np.asarray([0, 0]),
        alpha=1.0,
        center_donors=False,
        balance_donors=False,
    )
    maximum = np.finfo(np.float64).max

    with pytest.raises(ArithmeticError, match="matvec.*nonfinite"):
        operator.matvec(np.asarray([maximum, -maximum]))


@pytest.mark.parametrize("alpha", [0.0, -1.0, np.nan, np.inf, -np.inf, True])
def test_operator_rejects_invalid_alpha(alpha) -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()

    with pytest.raises(ValueError, match="alpha.*finite and strictly positive"):
        state.pflog_operator(sparse.csr_array(counts), chromosomes, donor_index, alpha=alpha)


@pytest.mark.parametrize(
    ("donor_index", "message"),
    [
        (np.asarray([0, 0, 1, 1], dtype=np.int64), "cell axis"),
        (np.asarray([[0, 1, 0, 1, 2]], dtype=np.int64), "one-dimensional"),
        (np.asarray([0.0, 1.0, 0.0, 1.0, 2.0]), "integer"),
        (np.asarray([False, False, True, True, True]), "integer"),
        (np.asarray([0, 1, 0, 1, -1], dtype=np.int64), "nonnegative"),
        (np.asarray([0, 2, 0, 2, 2], dtype=np.int64), "dense labels"),
    ],
)
def test_operator_rejects_invalid_donor_indices(donor_index: np.ndarray, message: str) -> None:
    state = _state_api()
    counts, chromosomes, _ = _operator_fixture()

    with pytest.raises((TypeError, ValueError), match=message):
        state.pflog_operator(sparse.csr_array(counts), chromosomes, donor_index, alpha=0.5)


@pytest.mark.parametrize("option", [0, 1, "yes", None])
def test_operator_requires_explicit_boolean_centering_and_balancing_options(option) -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()

    with pytest.raises(TypeError, match="center_donors.*boolean"):
        state.pflog_operator(
            sparse.csr_array(counts),
            chromosomes,
            donor_index,
            alpha=0.5,
            center_donors=option,
        )
    with pytest.raises(TypeError, match="balance_donors.*boolean"):
        state.pflog_operator(
            sparse.csr_array(counts),
            chromosomes,
            donor_index,
            alpha=0.5,
            balance_donors=option,
        )


def test_operator_rejects_exclusion_that_removes_every_gene() -> None:
    state = _state_api()

    with pytest.raises(ValueError, match="at least one active gene"):
        state.pflog_operator(
            _counts([[1, 0], [0, 2]]),
            np.asarray(["1", "1"]),
            np.asarray([0, 0]),
            alpha=0.5,
            excluded_chromosome="1",
        )


@pytest.mark.parametrize(
    ("method", "values", "message"),
    [
        ("matvec", np.zeros((4, 1)), "matvec input"),
        ("matvec", np.zeros(3), "matvec input"),
        ("rmatvec", np.zeros((5, 1)), "rmatvec input"),
        ("rmatvec", np.zeros(4), "rmatvec input"),
        ("matmat", np.zeros(4), "matmat input"),
        ("matmat", np.zeros((3, 2)), "matmat input"),
        ("rmatmat", np.zeros(5), "rmatmat input"),
        ("rmatmat", np.zeros((4, 2)), "rmatmat input"),
    ],
)
def test_operator_rejects_vector_and_block_shape_errors(method: str, values: np.ndarray, message: str) -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    operator = state.pflog_operator(sparse.csr_array(counts), chromosomes, donor_index, alpha=0.5)

    with pytest.raises(ValueError, match=message):
        getattr(operator, method)(values)


def test_operator_config_diagnostics_and_arrays_are_immutable() -> None:
    state = _state_api()
    counts, chromosomes, donor_index = _operator_fixture()
    operator = state.pflog_operator(sparse.csr_array(counts), chromosomes, donor_index, alpha=0.5)

    with pytest.raises(FrozenInstanceError):
        operator.config.alpha = 1.0
    with pytest.raises(FrozenInstanceError):
        operator.diagnostics.n_donors = 99
    with pytest.raises(ValueError, match="read-only"):
        operator.diagnostics.donor_counts[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        operator.diagnostics.cell_weights[0] = 99.0
