# pattern: Functional Core

from __future__ import annotations

import math

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from typing import Literal

import numpy as np

from scipy import sparse


_AUTOSOMES = tuple(str(chromosome) for chromosome in range(1, 23))
_SUPPORTED_CHROMOSOMES = frozenset((*_AUTOSOMES, "X", "Y", "MT"))
_MAX_EXACT_FLOAT64_INT = 2**53
_FLOAT64_EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class PFLogStatistics:
    r"""Immutable PFlog per-gene and chromosome-level sufficient statistics.

    This result contains only array and scalar numerical state. Non-autosomal
    genes (``X``, ``Y``, and ``MT``) are retained in the ``unassigned`` bin
    because LOCO exclusion is defined only for canonical autosomes.
    """

    n_cells: int
    n_genes: int
    gene_chromosomes: np.ndarray
    means: np.ndarray
    mean_error_scales: np.ndarray
    variances: np.ndarray
    variance_roundoff_tolerances: np.ndarray
    retained_gene_mask: np.ndarray
    a_terms: np.ndarray
    b_terms: np.ndarray
    autosome_numerators: np.ndarray
    autosome_denominators: np.ndarray
    autosome_gene_counts: np.ndarray
    unassigned_numerator: float
    unassigned_denominator: float
    unassigned_gene_count: int
    total_numerator: float
    total_denominator: float
    _total_numerator_abs_sum: float
    _total_denominator_abs_sum: float


@dataclass(frozen=True, slots=True)
class PFLogAlphaDiagnostics:
    r"""Immutable diagnostics for one genomewide or LOCO PFlog estimate."""

    alpha: float
    source: Literal["auto", "override"]
    excluded_chromosome: str | None
    retained_gene_count: int
    excluded_gene_count: int
    numerator: float
    denominator: float
    total_numerator: float
    total_denominator: float
    excluded_numerator: float
    excluded_denominator: float
    difference_numerator: float
    difference_denominator: float
    numerator_tolerance: float
    denominator_tolerance: float


def _readonly(values, *, dtype=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.flags.writeable = False
    return array


def _validate_counts(counts: sparse.csr_array) -> None:
    if not isinstance(counts, sparse.csr_array):
        raise TypeError("counts must be a canonical scipy.sparse.csr_array from the single-cell ingress boundary")
    if counts.ndim != 2 or counts.shape[1] == 0:
        raise ValueError("counts must be two-dimensional with at least one gene")
    if counts.shape[0] < 2:
        raise ValueError("PFlog statistics require at least two cells for ddof=1 variances")
    if not counts.has_canonical_format:
        raise ValueError("counts must have canonical sorted indices without duplicate coordinates")

    values = np.asarray(counts.data)
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(values.dtype, np.integer):
        raise TypeError("counts must retain non-boolean integer storage at the PFlog boundary")
    if np.issubdtype(values.dtype, np.signedinteger) and np.any(values < 0):
        raise ValueError("counts cannot contain negative values")
    if np.any(values > _MAX_EXACT_FLOAT64_INT):
        raise ValueError("counts must be at most 2**53 for exact float64 translation")


def _validate_chromosomes(gene_chromosomes: Sequence[str] | np.ndarray, n_genes: int) -> np.ndarray:
    chromosomes = np.asarray(gene_chromosomes)
    if chromosomes.ndim != 1 or chromosomes.shape[0] != n_genes:
        raise ValueError("gene_chromosomes must be one-dimensional and match the count-matrix gene axis")
    values = chromosomes.tolist()
    if any(not isinstance(value, str) for value in values):
        raise TypeError("gene_chromosomes must contain canonical string labels")
    unsupported = sorted({value for value in values if value not in _SUPPORTED_CHROMOSOMES})
    if unsupported:
        labels = ", ".join(repr(value) for value in unsupported)
        raise ValueError(f"gene_chromosomes contains noncanonical labels: {labels}")
    return _readonly(values, dtype=np.str_)


def _translated_gene_moments(values: np.ndarray, n_cells: int) -> tuple[float, float, float, float]:
    n_stored = values.size
    n_implicit = n_cells - n_stored
    anchor = int(values[0]) if n_implicit == 0 else 0
    differences = tuple(int(value) - anchor for value in values)
    int64 = np.iinfo(np.int64)
    if any(difference < int64.min or difference > int64.max for difference in differences):
        raise ArithmeticError("PFlog integer translation exceeded checked signed-integer range")

    stored_difference_sum = math.fsum(float(difference) for difference in differences)
    implicit_difference_sum = float(n_implicit * -anchor)
    mean_difference = math.fsum((stored_difference_sum, implicit_difference_sum)) / n_cells
    mean = math.fsum((float(anchor), mean_difference))

    centered_stored = ((float(difference) - mean_difference) ** 2 for difference in differences)
    implicit_delta = -float(anchor) - mean_difference
    implicit_sum_squares = float(n_implicit) * implicit_delta * implicit_delta
    sum_squares = math.fsum(chain(centered_stored, (implicit_sum_squares,)))
    variance = sum_squares / (n_cells - 1)
    variance_tolerance = 64.0 * _FLOAT64_EPS * max(sum_squares, 1.0) / (n_cells - 1)
    if variance < 0.0:
        if abs(variance) <= variance_tolerance:
            variance = 0.0
        else:
            raise ArithmeticError("PFlog variance is negative beyond its scale-aware float64 roundoff tolerance")

    exact_mean = Fraction(sum(int(value) for value in values), n_cells)
    representation_error = float(abs(Fraction.from_float(mean) - exact_mean))
    mean_error_scale = max(representation_error, 0.5 * math.ulp(mean))
    return mean, mean_error_scale, variance, variance_tolerance


def compute_pflog_statistics(
    counts: sparse.csr_array,
    gene_chromosomes: Sequence[str] | np.ndarray,
) -> PFLogStatistics:
    r"""Compute sparse PFlog sufficient statistics without dense materialization.

    **Arguments:**

    counts
        Canonical integer CSR counts with cells on rows and genes on columns.
    gene_chromosomes
        Canonical chromosome labels in count-matrix column order.

    **Returns:**

    Immutable float64 per-gene moments, through-origin contributions, and
    compensated chromosome bins used by strict LOCO estimation.

    **Raises:**

    TypeError
        If counts or chromosome labels violate the canonical ingress types.
    ValueError
        If the matrix has fewer than two cells or invalid shape, values, or
        chromosome labels.
    ArithmeticError
        If a translated variance violates the numerical roundoff contract.
    """
    _validate_counts(counts)
    n_cells, n_genes = counts.shape
    chromosomes = _validate_chromosomes(gene_chromosomes, n_genes)
    columns = counts.tocsc(copy=False)

    means = np.empty(n_genes, dtype=np.float64)
    mean_error_scales = np.empty(n_genes, dtype=np.float64)
    variances = np.empty(n_genes, dtype=np.float64)
    variance_tolerances = np.empty(n_genes, dtype=np.float64)
    for gene_index in range(n_genes):
        start = int(columns.indptr[gene_index])
        stop = int(columns.indptr[gene_index + 1])
        values = np.asarray(columns.data[start:stop])
        mean, mean_error_scale, variance, variance_tolerance = _translated_gene_moments(values, n_cells)
        means[gene_index] = mean
        mean_error_scales[gene_index] = mean_error_scale
        variances[gene_index] = variance
        variance_tolerances[gene_index] = variance_tolerance

    retained = np.isfinite(means) & np.isfinite(variances) & (means > 0.0) & (variances > 0.0)
    mean_squares = means * means
    a_terms = mean_squares * (variances - means)
    b_terms = mean_squares * mean_squares
    if not np.isfinite(a_terms).all() or not np.isfinite(b_terms).all():
        raise ArithmeticError("PFlog per-gene through-origin terms must be finite")

    numerator_bins: list[list[float]] = [[] for _ in _AUTOSOMES]
    denominator_bins: list[list[float]] = [[] for _ in _AUTOSOMES]
    unassigned_numerators: list[float] = []
    unassigned_denominators: list[float] = []
    autosome_gene_counts = np.zeros(len(_AUTOSOMES), dtype=np.int64)
    unassigned_gene_count = 0
    for gene_index in np.flatnonzero(retained):
        chromosome = str(chromosomes[gene_index])
        numerator = float(a_terms[gene_index])
        denominator = float(b_terms[gene_index])
        if chromosome in _AUTOSOMES:
            bin_index = int(chromosome) - 1
            numerator_bins[bin_index].append(numerator)
            denominator_bins[bin_index].append(denominator)
            autosome_gene_counts[bin_index] += 1
        else:
            unassigned_numerators.append(numerator)
            unassigned_denominators.append(denominator)
            unassigned_gene_count += 1

    autosome_numerators = np.asarray([math.fsum(values) for values in numerator_bins], dtype=np.float64)
    autosome_denominators = np.asarray([math.fsum(values) for values in denominator_bins], dtype=np.float64)
    unassigned_numerator = math.fsum(unassigned_numerators)
    unassigned_denominator = math.fsum(unassigned_denominators)
    retained_a_terms = a_terms[retained]
    retained_b_terms = b_terms[retained]
    total_numerator = math.fsum(float(value) for value in retained_a_terms)
    total_denominator = math.fsum(float(value) for value in retained_b_terms)

    return PFLogStatistics(
        n_cells=n_cells,
        n_genes=n_genes,
        gene_chromosomes=chromosomes,
        means=_readonly(means),
        mean_error_scales=_readonly(mean_error_scales),
        variances=_readonly(variances),
        variance_roundoff_tolerances=_readonly(variance_tolerances),
        retained_gene_mask=_readonly(retained),
        a_terms=_readonly(a_terms),
        b_terms=_readonly(b_terms),
        autosome_numerators=_readonly(autosome_numerators),
        autosome_denominators=_readonly(autosome_denominators),
        autosome_gene_counts=_readonly(autosome_gene_counts),
        unassigned_numerator=unassigned_numerator,
        unassigned_denominator=unassigned_denominator,
        unassigned_gene_count=unassigned_gene_count,
        total_numerator=total_numerator,
        total_denominator=total_denominator,
        _total_numerator_abs_sum=math.fsum(abs(float(value)) for value in retained_a_terms),
        _total_denominator_abs_sum=math.fsum(abs(float(value)) for value in retained_b_terms),
    )


def _validated_excluded_chromosome(excluded_chromosome: str | None) -> str | None:
    if excluded_chromosome is None:
        return None
    if not isinstance(excluded_chromosome, str) or excluded_chromosome not in _AUTOSOMES:
        raise ValueError("excluded_chromosome must be a canonical autosome string from '1' through '22'")
    return excluded_chromosome


def _validated_override(override: float | None) -> float | None:
    if override is None:
        return None
    if isinstance(override, (bool, np.bool_)):
        raise ValueError("PFlog alpha override must be finite and strictly positive")
    try:
        value = float(override)
    except (TypeError, ValueError) as error:
        raise ValueError("PFlog alpha override must be finite and strictly positive") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("PFlog alpha override must be finite and strictly positive")
    return value


def estimate_pflog_alpha(
    statistics: PFLogStatistics,
    *,
    excluded_chromosome: str | None = None,
    override: float | None = None,
) -> PFLogAlphaDiagnostics:
    r"""Estimate strict genomewide or LOCO PFlog alpha from cached statistics.

    **Arguments:**

    statistics
        Sufficient statistics returned by :func:`compute_pflog_statistics`.
    excluded_chromosome
        Optional canonical autosome label (``"1"`` through ``"22"``). An
        absent autosome is a valid no-op exclusion.
    override
        Optional finite, strictly positive alpha. A valid override preserves
        automatic-fit diagnostics while bypassing automatic validity failures.

    **Returns:**

    Immutable estimate provenance and direct-versus-subtractive accumulation
    diagnostics.

    **Raises:**

    TypeError
        If ``statistics`` was not produced by this PFlog statistics API.
    ValueError
        If the exclusion or override is invalid, or an automatic estimate has
        a nonfinite or nonpositive numerator, denominator, or alpha.
    ArithmeticError
        If direct LOCO accumulation disagrees with the total-minus-held-out
        diagnostic identity beyond float64 roundoff tolerance.
    """
    if not isinstance(statistics, PFLogStatistics):
        raise TypeError("statistics must be returned by compute_pflog_statistics")
    chromosome = _validated_excluded_chromosome(excluded_chromosome)
    override_value = _validated_override(override)

    if chromosome is None:
        excluded_index = None
        excluded_gene_count = 0
        excluded_numerator = 0.0
        excluded_denominator = 0.0
        retained_numerator_bins = statistics.autosome_numerators
        retained_denominator_bins = statistics.autosome_denominators
    else:
        excluded_index = int(chromosome) - 1
        excluded_gene_count = int(statistics.autosome_gene_counts[excluded_index])
        excluded_numerator = float(statistics.autosome_numerators[excluded_index])
        excluded_denominator = float(statistics.autosome_denominators[excluded_index])
        retained_numerator_bins = np.delete(statistics.autosome_numerators, excluded_index)
        retained_denominator_bins = np.delete(statistics.autosome_denominators, excluded_index)

    numerator = math.fsum(
        chain((float(value) for value in retained_numerator_bins), (statistics.unassigned_numerator,))
    )
    denominator = math.fsum(
        chain((float(value) for value in retained_denominator_bins), (statistics.unassigned_denominator,))
    )
    difference_numerator = statistics.total_numerator - excluded_numerator
    difference_denominator = statistics.total_denominator - excluded_denominator
    numerator_tolerance = 64.0 * _FLOAT64_EPS * max(1.0, statistics._total_numerator_abs_sum)
    denominator_tolerance = 64.0 * _FLOAT64_EPS * max(1.0, statistics._total_denominator_abs_sum)
    if abs(difference_numerator - numerator) > numerator_tolerance:
        raise ArithmeticError("PFlog LOCO numerator disagrees with total-minus-held-out beyond float64 tolerance")
    if abs(difference_denominator - denominator) > denominator_tolerance:
        raise ArithmeticError("PFlog LOCO denominator disagrees with total-minus-held-out beyond float64 tolerance")

    retained_gene_count = int(np.count_nonzero(statistics.retained_gene_mask)) - excluded_gene_count
    if override_value is None:
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("automatic PFlog denominator must be finite and strictly positive")
        if not math.isfinite(numerator) or numerator <= 0.0:
            raise ValueError("automatic PFlog numerator must be finite and strictly positive")
        alpha = numerator / denominator
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("automatic PFlog alpha must be finite and strictly positive")
        source: Literal["auto", "override"] = "auto"
    else:
        alpha = override_value
        source = "override"

    return PFLogAlphaDiagnostics(
        alpha=alpha,
        source=source,
        excluded_chromosome=chromosome,
        retained_gene_count=retained_gene_count,
        excluded_gene_count=excluded_gene_count,
        numerator=numerator,
        denominator=denominator,
        total_numerator=statistics.total_numerator,
        total_denominator=statistics.total_denominator,
        excluded_numerator=excluded_numerator,
        excluded_denominator=excluded_denominator,
        difference_numerator=difference_numerator,
        difference_denominator=difference_denominator,
        numerator_tolerance=numerator_tolerance,
        denominator_tolerance=denominator_tolerance,
    )
