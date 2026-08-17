import math

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from scipy import sparse

import jaxqtl


MAX_EXACT_FLOAT64_INT = 2**53


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


def test_pflog_statistics_api_is_available_from_state_package() -> None:
    state = _state_api()

    assert hasattr(state, "compute_pflog_statistics")
    assert hasattr(state, "estimate_pflog_alpha")
    assert state.__all__ == ["compute_pflog_statistics", "estimate_pflog_alpha"]


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
