# pattern: Functional Core

from __future__ import annotations

import math

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import final

import numpy as np

from scipy import sparse

from ._pflog import (
    _readonly,
    _validate_chromosomes,
    _validate_counts,
    _validated_excluded_chromosome,
)


@dataclass(frozen=True, slots=True)
class PFLogOperatorConfig:
    r"""Immutable effective configuration for a PFlog state operator."""

    alpha: float
    excluded_chromosome: str | None
    center_donors: bool
    balance_donors: bool


@dataclass(frozen=True, slots=True)
class PFLogOperatorDiagnostics:
    r"""Immutable dimensions, sparse-storage accounting, and donor diagnostics."""

    n_cells: int
    n_input_genes: int
    n_active_genes: int
    n_donors: int
    singleton_donor_count: int
    input_nnz: int
    transformed_nnz: int
    transformed_shape: tuple[int, int]
    active_gene_indices: np.ndarray
    donor_counts: np.ndarray
    cell_weights: np.ndarray


def _validated_alpha(alpha: float) -> float:
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("PFlog operator alpha must be finite and strictly positive")
    try:
        value = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("PFlog operator alpha must be finite and strictly positive") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("PFlog operator alpha must be finite and strictly positive")
    return value


def _validated_boolean(value: bool, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an explicit boolean")
    return bool(value)


def _validated_donor_index(donor_index: Sequence[int] | np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    donors = np.asarray(donor_index)
    if donors.ndim != 1:
        raise ValueError("donor_index must be one-dimensional")
    if donors.shape[0] != n_cells:
        raise ValueError("donor_index must match the count-matrix cell axis")
    if np.issubdtype(donors.dtype, np.bool_) or not np.issubdtype(donors.dtype, np.integer):
        raise TypeError("donor_index must contain non-boolean integer labels")
    if np.issubdtype(donors.dtype, np.signedinteger) and np.any(donors < 0):
        raise ValueError("donor_index labels must be nonnegative")

    unique_donors = np.unique(donors)
    expected_donors = np.arange(unique_donors.size, dtype=unique_donors.dtype)
    if not np.array_equal(unique_donors, expected_donors):
        raise ValueError("donor_index must use dense labels covering every integer in [0, N)")
    normalized = np.asarray(donors, dtype=np.intp)
    donor_counts = np.bincount(normalized, minlength=unique_donors.size).astype(np.int64, copy=False)
    return _readonly(normalized.copy()), _readonly(donor_counts.copy())


def _transformed_sparse_counts(
    counts: sparse.csr_array,
    active_gene_indices: np.ndarray,
    alpha: float,
) -> sparse.csr_array:
    selected = sparse.csr_array(counts[:, active_gene_indices], copy=True)
    source_values = np.asarray(selected.data)
    transformed_values = np.zeros(source_values.shape, dtype=np.float64)
    positive = source_values > 0
    if np.any(positive):
        log_product = math.log(4.0) + math.log(alpha) + np.log(source_values[positive].astype(np.float64))
        transformed_values[positive] = np.logaddexp(0.0, log_product)
    if not np.isfinite(transformed_values).all():
        raise ArithmeticError("PFlog transformed sparse values must be finite")

    transformed = sparse.csr_array(
        (
            transformed_values,
            np.asarray(selected.indices).copy(),
            np.asarray(selected.indptr).copy(),
        ),
        shape=selected.shape,
        copy=False,
    )
    transformed.data.flags.writeable = False
    transformed.indices.flags.writeable = False
    transformed.indptr.flags.writeable = False
    return transformed


def _validated_vector(values, *, size: int, method: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape != (size,):
        raise ValueError(f"{method} input must have shape ({size},); received {array.shape}")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{method} input must contain real numerical values")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{method} input must contain real numerical values")
    return np.asarray(array, dtype=np.float64)


def _validated_matrix(values, *, rows: int, method: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] != rows:
        raise ValueError(f"{method} input must have shape ({rows}, k); received {array.shape}")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{method} input must contain real numerical values")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{method} input must contain real numerical values")
    return np.asarray(array, dtype=np.float64)


@final
@dataclass(frozen=True, slots=True)
class PFLogOperator:
    r"""Implicit donor-balanced PFlog operator with exact transpose actions.

    The represented matrix is ``sqrt(D_w) C_D L J_q``. Only the transformed
    nonzero values of ``L`` are stored; feature and donor centers are applied
    through reductions during each action.
    """

    config: PFLogOperatorConfig
    diagnostics: PFLogOperatorDiagnostics
    _transformed_counts: sparse.csr_array = field(repr=False)
    _donor_index: np.ndarray = field(repr=False)
    _sqrt_cell_weights: np.ndarray = field(repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        r"""Return the cell-by-active-gene operator dimensions."""
        return self._transformed_counts.shape

    @property
    def dtype(self) -> np.dtype:
        r"""Return the fixed float64 action dtype."""
        return np.dtype(np.float64)

    def _feature_center(self, values: np.ndarray) -> np.ndarray:
        return values - np.mean(values, axis=0)

    def _donor_center(self, values: np.ndarray) -> np.ndarray:
        if not self.config.center_donors:
            return values
        donor_counts = self.diagnostics.donor_counts
        if values.ndim == 1:
            totals = np.bincount(
                self._donor_index,
                weights=values,
                minlength=self.diagnostics.n_donors,
            )
            donor_means = totals / donor_counts
        else:
            totals = np.zeros((self.diagnostics.n_donors, values.shape[1]), dtype=np.float64)
            np.add.at(totals, self._donor_index, values)
            donor_means = totals / donor_counts[:, None]
        return values - donor_means[self._donor_index]

    def _balance(self, values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return self._sqrt_cell_weights * values
        return self._sqrt_cell_weights[:, None] * values

    def matvec(self, vector) -> np.ndarray:
        r"""Apply the PFlog operator to one active-gene vector.

        **Arguments:**

        vector
            Real vector with shape ``(n_active_genes,)``.

        **Returns:**

        A float64 cell vector with shape ``(n_cells,)``.

        **Raises:**

        ValueError
            If the input shape is incompatible with the operator.
        TypeError
            If the input is not real numerical data.
        """
        values = _validated_vector(vector, size=self.shape[1], method="matvec")
        cells = np.asarray(self._transformed_counts @ self._feature_center(values), dtype=np.float64)
        return self._balance(self._donor_center(cells))

    def rmatvec(self, vector) -> np.ndarray:
        r"""Apply the exact transpose to one cell vector.

        **Arguments:**

        vector
            Real vector with shape ``(n_cells,)``.

        **Returns:**

        A float64 active-gene vector with shape ``(n_active_genes,)``.

        **Raises:**

        ValueError
            If the input shape is incompatible with the operator.
        TypeError
            If the input is not real numerical data.
        """
        values = _validated_vector(vector, size=self.shape[0], method="rmatvec")
        centered = self._donor_center(self._balance(values))
        features = np.asarray(self._transformed_counts.T @ centered, dtype=np.float64)
        return self._feature_center(features)

    def matmat(self, matrix) -> np.ndarray:
        r"""Apply the PFlog operator to a block of active-gene vectors.

        **Arguments:**

        matrix
            Real matrix with shape ``(n_active_genes, k)``.

        **Returns:**

        A float64 cell block with shape ``(n_cells, k)``.

        **Raises:**

        ValueError
            If the input shape is incompatible with the operator.
        TypeError
            If the input is not real numerical data.
        """
        values = _validated_matrix(matrix, rows=self.shape[1], method="matmat")
        cells = np.asarray(self._transformed_counts @ self._feature_center(values), dtype=np.float64)
        return self._balance(self._donor_center(cells))

    def rmatmat(self, matrix) -> np.ndarray:
        r"""Apply the exact transpose to a block of cell vectors.

        **Arguments:**

        matrix
            Real matrix with shape ``(n_cells, k)``.

        **Returns:**

        A float64 active-gene block with shape ``(n_active_genes, k)``.

        **Raises:**

        ValueError
            If the input shape is incompatible with the operator.
        TypeError
            If the input is not real numerical data.
        """
        values = _validated_matrix(matrix, rows=self.shape[0], method="rmatmat")
        centered = self._donor_center(self._balance(values))
        features = np.asarray(self._transformed_counts.T @ centered, dtype=np.float64)
        return self._feature_center(features)


def pflog_operator(
    counts: sparse.csr_array,
    gene_chromosomes: Sequence[str] | np.ndarray,
    donor_index: Sequence[int] | np.ndarray,
    *,
    alpha: float,
    excluded_chromosome: str | None = None,
    center_donors: bool = True,
    balance_donors: bool = True,
) -> PFLogOperator:
    r"""Construct an implicit PFlog, CLR, donor-center, and balance operator.

    **Arguments:**

    counts
        Canonical integer CSR counts with cells on rows and genes on columns.
    gene_chromosomes
        Canonical chromosome labels in count-matrix column order.
    donor_index
        Dense nonnegative integer donor labels in cell order. Labels may be
        nonsorted but must cover every integer in ``[0, N)``.
    alpha
        Finite strictly positive PFlog scale.
    excluded_chromosome
        Optional canonical autosome removed before PFlog transformation and
        CLR feature centering.
    center_donors
        Whether to apply within-donor cell centering. Defaults to ``True``.
    balance_donors
        Whether to apply equal-donor cell weights. Defaults to ``True``.

    **Returns:**

    A float64 implicit operator with vector, transpose-vector, and block
    actions. The returned diagnostics record the active genes, donor counts,
    cell weights, and sparse-storage dimensions.

    **Raises:**

    TypeError
        If counts, chromosomes, donor labels, or boolean options violate the
        canonical boundary types.
    ValueError
        If alpha, shapes, chromosome exclusion, or donor label coverage is
        invalid, including exclusion of every gene.
    ArithmeticError
        If a finite valid input unexpectedly produces a nonfinite transform.
    """
    _validate_counts(counts)
    n_cells, n_input_genes = counts.shape
    chromosomes = _validate_chromosomes(gene_chromosomes, n_input_genes)
    chromosome = _validated_excluded_chromosome(excluded_chromosome)
    alpha_value = _validated_alpha(alpha)
    center = _validated_boolean(center_donors, name="center_donors")
    balance = _validated_boolean(balance_donors, name="balance_donors")
    donors, donor_counts = _validated_donor_index(donor_index, n_cells)

    active_mask = np.ones(n_input_genes, dtype=np.bool_)
    if chromosome is not None:
        active_mask = chromosomes != chromosome
    active_gene_indices = np.flatnonzero(active_mask).astype(np.int64, copy=False)
    if active_gene_indices.size == 0:
        raise ValueError("PFlog operator requires at least one active gene after chromosome exclusion")

    transformed = _transformed_sparse_counts(counts, active_gene_indices, alpha_value)
    n_donors = donor_counts.size
    cell_weights = np.ones(n_cells, dtype=np.float64)
    if balance:
        donor_weights = np.zeros(n_donors, dtype=np.float64)
        nonsingleton = donor_counts >= 2
        donor_weights[nonsingleton] = 1.0 / (n_donors * (donor_counts[nonsingleton] - 1))
        cell_weights = donor_weights[donors]
    readonly_cell_weights = _readonly(cell_weights.copy())
    sqrt_cell_weights = _readonly(np.sqrt(readonly_cell_weights))

    config = PFLogOperatorConfig(
        alpha=alpha_value,
        excluded_chromosome=chromosome,
        center_donors=center,
        balance_donors=balance,
    )
    diagnostics = PFLogOperatorDiagnostics(
        n_cells=n_cells,
        n_input_genes=n_input_genes,
        n_active_genes=active_gene_indices.size,
        n_donors=n_donors,
        singleton_donor_count=int(np.count_nonzero(donor_counts == 1)),
        input_nnz=counts.nnz,
        transformed_nnz=transformed.nnz,
        transformed_shape=transformed.shape,
        active_gene_indices=_readonly(active_gene_indices.copy()),
        donor_counts=donor_counts,
        cell_weights=readonly_cell_weights,
    )
    return PFLogOperator(
        config=config,
        diagnostics=diagnostics,
        _transformed_counts=transformed,
        _donor_index=donors,
        _sqrt_cell_weights=sqrt_cell_weights,
    )
