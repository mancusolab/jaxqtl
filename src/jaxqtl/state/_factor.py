# pattern: Functional Core

from __future__ import annotations

import math

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import final, Literal

import numpy as np

from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from ._operator import _operator_with_balancing, pflog_operator, PFLogOperator
from ._pflog import (
    _AUTOSOMES,
    _readonly,
    _validated_excluded_chromosome,
    compute_pflog_statistics,
    estimate_pflog_alpha,
    PFLogStatistics,
)


_FLOAT64_EPS = float(np.finfo(np.float64).eps)
_FLOAT64_TINY = float(np.finfo(np.float64).tiny)
_MAX_UINT64 = 2**64 - 1


@dataclass(frozen=True, slots=True)
class StateFactorDiagnostics:
    r"""Immutable convergence, scaling, and provenance diagnostics.

    The public result contains only Python scalars and read-only NumPy arrays;
    SciPy operators and solver objects remain private implementation details.
    """

    alpha: float
    alpha_source: Literal["auto", "override"]
    alpha_retained_gene_count: int
    alpha_excluded_gene_count: int
    alpha_numerator: float
    alpha_denominator: float
    alpha_excluded_numerator: float
    alpha_excluded_denominator: float
    excluded_chromosome: str | None
    active_gene_indices: np.ndarray
    rank: int
    solver: Literal["propack", "arpack"]
    seed: int
    tol: float
    maxiter: int
    propack_kmax: int | None
    arpack_ncv: int | None
    sigma_floor: float
    residual_limit: float
    max_forward_residual: float
    max_adjoint_residual: float
    loading_orthogonality_error: float
    singular_values: np.ndarray
    donor_counts: np.ndarray
    singleton_donor_count: int
    center_within_donor: bool
    balance_donors: bool
    operator_shape: tuple[int, int]
    transformed_shape: tuple[int, int]
    transformed_nnz: int
    factors_shape: tuple[int, int]
    loadings_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class StateFactorResult:
    r"""Validated donor-centered state factors and active-gene loadings.

    ``factors`` are the natural unweighted coordinates $S=ZV_r$ rather than
    weighted or whitened left singular scores. All arrays are read-only.
    """

    factors: np.ndarray
    loadings: np.ndarray
    singular_values: np.ndarray
    diagnostics: StateFactorDiagnostics


def _validated_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-boolean integer")
    normalized = int(value)
    if normalized < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


def _validated_solver(solver: str) -> Literal["propack", "arpack"]:
    if solver == "propack":
        return "propack"
    if solver == "arpack":
        return "arpack"
    raise ValueError("solver must be explicitly selected as 'propack' or 'arpack'")


def _validated_tol(tol: float) -> float:
    if isinstance(tol, (bool, np.bool_)):
        raise ValueError("tol must be finite and strictly positive")
    try:
        normalized = float(tol)
    except (TypeError, ValueError) as error:
        raise ValueError("tol must be finite and strictly positive") from error
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("tol must be finite and strictly positive")
    return normalized


def _validated_seed(seed: int) -> int:
    normalized = _validated_integer(seed, name="seed")
    if normalized > _MAX_UINT64:
        raise ValueError("seed must fit in an unsigned 64-bit integer")
    return normalized


def _validated_solver_configuration(
    *,
    solver: Literal["propack", "arpack"],
    rank: int,
    min_dimension: int,
    maxiter: int,
    ncv: int | None,
) -> int | None:
    if rank >= min_dimension:
        raise ValueError("rank must be strictly smaller than min(n_cells, n_active_genes) for truncated factorization")
    if solver == "propack":
        if ncv is not None:
            raise ValueError("ncv is allowed only for the ARPACK solver")
        if maxiter < rank:
            raise ValueError("PROPACK maxiter is its Krylov dimension and must be at least rank")
        return None

    if ncv is None:
        raise ValueError("ncv is required for the ARPACK solver")
    normalized_ncv = _validated_integer(ncv, name="ncv", minimum=1)
    if not rank < normalized_ncv < min_dimension:
        raise ValueError("ARPACK ncv must satisfy rank < ncv < min(n_cells, n_active_genes)")
    return normalized_ncv


def _validated_alpha_override(pflog_alpha: str | float) -> float | None:
    if isinstance(pflog_alpha, str):
        if pflog_alpha != "auto":
            raise ValueError("pflog_alpha must be 'auto' or a finite strictly positive override")
        return None
    return pflog_alpha


@final
class _StateLinearOperator(sparse_linalg.LinearOperator):
    """Private SciPy adapter over the backend-neutral domain operator."""

    def __init__(self, operator: PFLogOperator) -> None:
        self._operator = operator
        super().__init__(dtype=np.dtype(np.float64), shape=operator.shape)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self._operator.matvec(x)

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return self._operator.rmatvec(x)

    def _matmat(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return self._operator.matmat(X)

    def _rmatmat(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return self._operator.rmatmat(X)


def _solve_right_triplets(
    operator: PFLogOperator,
    *,
    rank: int,
    solver: Literal["propack", "arpack"],
    tol: float,
    maxiter: int,
    seed: int,
    ncv: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    scipy_operator = _StateLinearOperator(operator)
    try:
        solution = sparse_linalg.svds(
            scipy_operator,
            k=rank,
            ncv=ncv,
            tol=tol,
            which="LM",
            maxiter=maxiter,
            return_singular_vectors="vh",
            solver=solver,
            rng=np.random.default_rng(seed),
        )
    except sparse_linalg.ArpackNoConvergence as error:
        raise RuntimeError("ARPACK failed to converge; partial singular triplets were discarded") from error
    except np.linalg.LinAlgError as error:
        label = solver.upper()
        raise RuntimeError(f"{label} factorization failed; no partial result was returned") from error

    if not isinstance(solution, tuple) or len(solution) != 3:
        raise RuntimeError("SciPy returned an incomplete truncated-SVD result")
    _, singular_values_raw, vh_raw = solution
    singular_values = np.asarray(singular_values_raw, dtype=np.float64)
    vh = None if vh_raw is None else np.asarray(vh_raw, dtype=np.float64)
    if singular_values.shape != (rank,) or vh is None or vh.shape != (rank, operator.shape[1]):
        raise RuntimeError("SciPy returned incomplete requested singular triplets")
    return singular_values, vh


def _validated_triplets(
    operator: PFLogOperator,
    singular_values: np.ndarray,
    vh: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    order = np.argsort(singular_values, kind="stable")[::-1]
    singular_values = np.asarray(singular_values[order], dtype=np.float64)
    loadings = np.asarray(vh[order].T, dtype=np.float64)
    if not np.isfinite(singular_values).all():
        raise RuntimeError("solver returned nonfinite singular values")
    if not np.isfinite(loadings).all():
        raise RuntimeError("solver returned nonfinite right singular vectors")

    sigma_max = float(singular_values[0])
    if not math.isfinite(sigma_max) or sigma_max <= 0.0:
        raise RuntimeError("largest requested singular value must be finite and strictly positive")
    dmax = max(operator.shape)
    sigma_floor = max(tol, _FLOAT64_EPS * dmax) * sigma_max
    if np.any(singular_values <= sigma_floor):
        raise RuntimeError("requested singular triplet is at or below the scale-aware sigma floor")

    balanced_scores = operator.matmat(loadings)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        left_vectors = balanced_scores / singular_values[None, :]
    if not np.isfinite(left_vectors).all():
        raise RuntimeError("reconstructed balanced left singular vectors are nonfinite")
    adjoint_scores = operator.rmatmat(left_vectors)

    forward_numerators = np.linalg.norm(balanced_scores - left_vectors * singular_values[None, :], axis=0)
    forward_denominators = np.maximum.reduce(
        (
            singular_values,
            np.linalg.norm(balanced_scores, axis=0),
            np.full(singular_values.shape, _FLOAT64_TINY),
        )
    )
    adjoint_numerators = np.linalg.norm(adjoint_scores - loadings * singular_values[None, :], axis=0)
    adjoint_denominators = np.maximum.reduce(
        (
            singular_values,
            np.linalg.norm(adjoint_scores, axis=0),
            np.full(singular_values.shape, _FLOAT64_TINY),
        )
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        forward_residuals = forward_numerators / forward_denominators
        adjoint_residuals = adjoint_numerators / adjoint_denominators
    gram_error = loadings.T @ loadings - np.eye(loadings.shape[1], dtype=np.float64)
    orthogonality_error = float(np.linalg.norm(gram_error, ord=2))
    max_forward_residual = float(np.max(forward_residuals))
    max_adjoint_residual = float(np.max(adjoint_residuals))
    residual_limit = max(10.0 * tol, 100.0 * _FLOAT64_EPS * dmax)
    diagnostics = (max_forward_residual, max_adjoint_residual, orthogonality_error)
    if not all(math.isfinite(value) for value in diagnostics):
        raise RuntimeError("singular-triplet residual diagnostics must be finite")
    if max_forward_residual > residual_limit or max_adjoint_residual > residual_limit:
        raise RuntimeError("singular-triplet residual exceeds the scale-aware residual limit")
    if orthogonality_error > residual_limit:
        raise RuntimeError("right-loading orthogonality error exceeds the scale-aware residual limit")
    return (
        singular_values,
        loadings,
        sigma_floor,
        residual_limit,
        max_forward_residual,
        max_adjoint_residual,
        orthogonality_error,
    )


def _canonicalize_signs(loadings: np.ndarray, factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    canonical_loadings = loadings.copy()
    canonical_factors = factors.copy()
    for column in range(canonical_loadings.shape[1]):
        anchor = int(np.argmax(np.abs(canonical_loadings[:, column])))
        if canonical_loadings[anchor, column] < 0.0:
            canonical_loadings[:, column] *= -1.0
            canonical_factors[:, column] *= -1.0
    return canonical_loadings, canonical_factors


def _construct_from_statistics(
    counts: sparse.csr_array,
    donor_index: Sequence[int] | np.ndarray,
    statistics: PFLogStatistics,
    *,
    rank: int,
    solver: Literal["propack", "arpack"],
    tol: float,
    maxiter: int,
    seed: int,
    ncv: int | None,
    exclude_chromosome: str | None,
    pflog_alpha: str | float,
    center_within_donor: bool,
    balance_donors: bool,
) -> StateFactorResult:
    chromosome = _validated_excluded_chromosome(exclude_chromosome)
    active_gene_count = statistics.n_genes
    if chromosome is not None:
        active_gene_count = int(np.count_nonzero(statistics.gene_chromosomes != chromosome))
    min_dimension = min(statistics.n_cells, active_gene_count)
    resolved_ncv = _validated_solver_configuration(
        solver=solver,
        rank=rank,
        min_dimension=min_dimension,
        maxiter=maxiter,
        ncv=ncv,
    )
    alpha_diagnostics = estimate_pflog_alpha(
        statistics,
        excluded_chromosome=chromosome,
        override=_validated_alpha_override(pflog_alpha),
    )
    base_operator = pflog_operator(
        counts,
        statistics.gene_chromosomes,
        donor_index,
        alpha=alpha_diagnostics.alpha,
        excluded_chromosome=chromosome,
        center_donors=center_within_donor,
        balance_donors=balance_donors,
    )
    singular_values_raw, vh = _solve_right_triplets(
        base_operator,
        rank=rank,
        solver=solver,
        tol=tol,
        maxiter=maxiter,
        seed=seed,
        ncv=resolved_ncv,
    )
    (
        singular_values,
        loadings,
        sigma_floor,
        residual_limit,
        max_forward_residual,
        max_adjoint_residual,
        orthogonality_error,
    ) = _validated_triplets(base_operator, singular_values_raw, vh, tol=tol)

    unweighted_operator = _operator_with_balancing(base_operator, balance_donors=False)
    factors = unweighted_operator.matmat(loadings)
    if not np.isfinite(factors).all():
        raise RuntimeError("unweighted state factors must be finite")
    loadings, factors = _canonicalize_signs(loadings, factors)
    readonly_singular_values = _readonly(singular_values.copy())
    readonly_loadings = _readonly(loadings.copy())
    readonly_factors = _readonly(factors.copy())
    operator_diagnostics = base_operator.diagnostics
    diagnostics = StateFactorDiagnostics(
        alpha=alpha_diagnostics.alpha,
        alpha_source=alpha_diagnostics.source,
        alpha_retained_gene_count=alpha_diagnostics.retained_gene_count,
        alpha_excluded_gene_count=alpha_diagnostics.excluded_gene_count,
        alpha_numerator=alpha_diagnostics.numerator,
        alpha_denominator=alpha_diagnostics.denominator,
        alpha_excluded_numerator=alpha_diagnostics.excluded_numerator,
        alpha_excluded_denominator=alpha_diagnostics.excluded_denominator,
        excluded_chromosome=alpha_diagnostics.excluded_chromosome,
        active_gene_indices=_readonly(operator_diagnostics.active_gene_indices.copy()),
        rank=rank,
        solver=solver,
        seed=seed,
        tol=tol,
        maxiter=maxiter,
        propack_kmax=maxiter if solver == "propack" else None,
        arpack_ncv=resolved_ncv,
        sigma_floor=sigma_floor,
        residual_limit=residual_limit,
        max_forward_residual=max_forward_residual,
        max_adjoint_residual=max_adjoint_residual,
        loading_orthogonality_error=orthogonality_error,
        singular_values=_readonly(singular_values.copy()),
        donor_counts=_readonly(operator_diagnostics.donor_counts.copy()),
        singleton_donor_count=operator_diagnostics.singleton_donor_count,
        center_within_donor=base_operator.config.center_donors,
        balance_donors=base_operator.config.balance_donors,
        operator_shape=base_operator.shape,
        transformed_shape=operator_diagnostics.transformed_shape,
        transformed_nnz=operator_diagnostics.transformed_nnz,
        factors_shape=readonly_factors.shape,
        loadings_shape=readonly_loadings.shape,
    )
    return StateFactorResult(
        factors=readonly_factors,
        loadings=readonly_loadings,
        singular_values=readonly_singular_values,
        diagnostics=diagnostics,
    )


def construct_state_factor(
    counts: sparse.csr_array,
    donor_index: Sequence[int] | np.ndarray,
    gene_chromosomes: Sequence[str] | np.ndarray,
    *,
    rank: int,
    solver: Literal["propack", "arpack"],
    tol: float,
    maxiter: int,
    seed: int,
    ncv: int | None = None,
    exclude_chromosome: str | None = None,
    pflog_alpha: str | float = "auto",
    center_within_donor: bool = True,
    balance_donors: bool = True,
) -> StateFactorResult:
    r"""Construct one deterministic matrix-free donor-balanced state factorization.

    **Arguments:**

    counts
        Canonical integer CSR counts with cells on rows and genes on columns.
    donor_index
        Dense zero-based donor labels in cell order.
    gene_chromosomes
        Canonical chromosome labels in gene order.
    rank
        Requested strict truncated rank.
    solver
        Explicit SciPy backend, either ``"propack"`` or ``"arpack"``.
    tol
        Finite strictly positive relative solver tolerance.
    maxiter
        Positive ARPACK iteration cap or PROPACK Krylov dimension.
    seed
        Unsigned 64-bit seed for deterministic solver initialization.
    ncv
        Required ARPACK Krylov subspace size and forbidden for PROPACK.
    exclude_chromosome
        Optional canonical autosome excluded before transformation and CLR.
    pflog_alpha
        ``"auto"`` for strict moment fitting or a positive finite override.
    center_within_donor
        Whether to center transformed features within donors.
    balance_donors
        Whether to learn loadings from the donor-balanced operator.

    **Returns:**

    Read-only unweighted factors, active-gene loadings, singular values, and
    backend-neutral diagnostics. Repeated singular values identify a subspace;
    only distinct loading columns have a stable individual orientation.

    **Raises:**

    TypeError
        If canonical array or integer/boolean contracts are violated.
    ValueError
        If alpha, rank, solver, tolerance, or solver dimensions are invalid.
    RuntimeError
        If SciPy fails, returns incomplete triplets, or violates the requested
        scale-aware singular-value, residual, or orthogonality contracts.
    """
    normalized_rank = _validated_integer(rank, name="rank", minimum=1)
    normalized_solver = _validated_solver(solver)
    normalized_tol = _validated_tol(tol)
    normalized_maxiter = _validated_integer(maxiter, name="maxiter", minimum=1)
    normalized_seed = _validated_seed(seed)
    statistics = compute_pflog_statistics(counts, gene_chromosomes)
    return _construct_from_statistics(
        counts,
        donor_index,
        statistics,
        rank=normalized_rank,
        solver=normalized_solver,
        tol=normalized_tol,
        maxiter=normalized_maxiter,
        seed=normalized_seed,
        ncv=ncv,
        exclude_chromosome=exclude_chromosome,
        pflog_alpha=pflog_alpha,
        center_within_donor=center_within_donor,
        balance_donors=balance_donors,
    )


def iter_loco_state_factors(
    counts: sparse.csr_array,
    donor_index: Sequence[int] | np.ndarray,
    gene_chromosomes: Sequence[str] | np.ndarray,
    *,
    rank: int,
    solver: Literal["propack", "arpack"],
    tol: float,
    maxiter: int,
    seed: int,
    ncv: int | None = None,
    chromosomes: Sequence[str] = _AUTOSOMES,
    pflog_alpha: str | float = "auto",
    center_within_donor: bool = True,
    balance_donors: bool = True,
) -> Iterator[StateFactorResult]:
    r"""Yield chromosome-specific state factors in caller-specified order.

    **Arguments:**

    counts
        Canonical integer CSR counts with cells on rows and genes on columns.
    donor_index
        Dense zero-based donor labels in cell order.
    gene_chromosomes
        Canonical chromosome labels in gene order.
    rank, solver, tol, maxiter, seed, ncv, pflog_alpha
        The same explicit numerical configuration as
        :func:`construct_state_factor`.
    chromosomes
        Ordered canonical autosomes to exclude. Defaults to ``"1"`` through
        ``"22"`` and preserves the supplied order exactly.
    center_within_donor, balance_donors
        Operator centering and loading-learning policies.

    **Returns:**

    A streaming iterator of complete read-only results. PFlog sufficient
    statistics are computed once and reused for every strict fast LOCO alpha.

    **Raises:**

    TypeError
        If inputs violate canonical boundary types.
    ValueError
        If chromosome or numerical configuration is invalid.
    RuntimeError
        If any requested factorization fails validation; no partial result is
        returned for that chromosome.
    """
    normalized_rank = _validated_integer(rank, name="rank", minimum=1)
    normalized_solver = _validated_solver(solver)
    normalized_tol = _validated_tol(tol)
    normalized_maxiter = _validated_integer(maxiter, name="maxiter", minimum=1)
    normalized_seed = _validated_seed(seed)
    if isinstance(chromosomes, (str, bytes)):
        raise TypeError("chromosomes must be an ordered sequence of canonical autosome strings")
    ordered_chromosomes = tuple(chromosomes)
    if not ordered_chromosomes:
        raise ValueError("chromosomes must contain at least one canonical autosome")

    statistics = compute_pflog_statistics(counts, gene_chromosomes)
    for chromosome in ordered_chromosomes:
        yield _construct_from_statistics(
            counts,
            donor_index,
            statistics,
            rank=normalized_rank,
            solver=normalized_solver,
            tol=normalized_tol,
            maxiter=normalized_maxiter,
            seed=normalized_seed,
            ncv=ncv,
            exclude_chromosome=chromosome,
            pflog_alpha=pflog_alpha,
            center_within_donor=center_within_donor,
            balance_donors=balance_donors,
        )
