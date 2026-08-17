# pattern: Functional Core

import inspect

from dataclasses import fields, FrozenInstanceError
from types import SimpleNamespace
from typing import cast, TYPE_CHECKING

import numpy as np
import pytest

from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator

import jaxqtl


if TYPE_CHECKING:
    from jaxqtl.state import StateFactorResult


class _DenseConversionForbiddenCSR(sparse.csr_array):
    def toarray(self, *args, **kwargs):
        raise AssertionError("state factorization must not densify counts")

    def todense(self, *args, **kwargs):
        raise AssertionError("state factorization must not densify counts")

    def __array__(self, *args, **kwargs):
        raise AssertionError("state factorization must not coerce sparse counts to a dense array")


class _IdentityDomainOperator:
    shape = (8, 6)
    config = SimpleNamespace(center_donors=True, balance_donors=True)
    diagnostics = SimpleNamespace(
        n_input_genes=6,
        n_active_genes=6,
        active_gene_indices=np.arange(6, dtype=np.int64),
        donor_counts=np.asarray([3, 3, 1, 1], dtype=np.int64),
        singleton_donor_count=2,
        transformed_shape=(8, 6),
        transformed_nnz=6,
    )

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        result = np.zeros(self.shape[0], dtype=np.float64)
        result[: self.shape[1]] = vector
        return result

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector[: self.shape[1]], dtype=np.float64)

    def matmat(self, matrix: np.ndarray) -> np.ndarray:
        result = np.zeros((self.shape[0], matrix.shape[1]), dtype=np.float64)
        result[: self.shape[1]] = matrix
        return result

    def rmatmat(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix[: self.shape[1]], dtype=np.float64)


def _state_api():
    assert hasattr(jaxqtl.state, "construct_state_factor"), "state factorization API is not available"
    assert hasattr(jaxqtl.state, "iter_loco_state_factors")
    assert hasattr(jaxqtl.state, "StateFactorDiagnostics")
    assert hasattr(jaxqtl.state, "StateFactorResult")
    return jaxqtl.state


def _fixture() -> tuple[sparse.csr_array, np.ndarray, np.ndarray]:
    counts = sparse.csr_array(
        np.asarray(
            [
                [0, 0, 1, 0, 2, 0],
                [0, 1, 0, 0, 4, 1],
                [6, 0, 1, 2, 0, 0],
                [0, 5, 0, 1, 0, 2],
                [1, 0, 8, 0, 2, 0],
                [0, 2, 0, 7, 0, 3],
                [9, 0, 2, 0, 5, 0],
                [0, 7, 0, 3, 0, 6],
            ],
            dtype=np.int64,
        )
    )
    donor_index = np.asarray([0, 0, 0, 1, 1, 1, 2, 3], dtype=np.int64)
    chromosomes = np.asarray(["1", "2", "3", "X", "1", "2"])
    return counts, donor_index, chromosomes


def _dense_operator(
    counts: sparse.csr_array,
    donor_index: np.ndarray,
    chromosomes: np.ndarray,
    *,
    alpha: float,
    exclude_chromosome: str | None,
    balance_donors: bool,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.ones(counts.shape[1], dtype=np.bool_)
    if exclude_chromosome is not None:
        active = chromosomes != exclude_chromosome
    dense = np.asarray(sparse.csr_array(counts[:, active]).toarray(), dtype=np.float64)
    transformed = np.log1p(4.0 * alpha * dense)
    transformed -= np.mean(transformed, axis=1, keepdims=True)
    for donor in range(int(np.max(donor_index)) + 1):
        rows = donor_index == donor
        transformed[rows] -= np.mean(transformed[rows], axis=0, keepdims=True)
    if balance_donors:
        donor_counts = np.bincount(donor_index)
        weights = np.zeros(donor_index.size, dtype=np.float64)
        for donor, count in enumerate(donor_counts):
            if count >= 2:
                weights[donor_index == donor] = 1.0 / (donor_counts.size * (count - 1))
        transformed = np.sqrt(weights)[:, None] * transformed
    return transformed, np.flatnonzero(active)


def _construct(*, solver: str = "propack", rank: int = 2, **kwargs):
    state = _state_api()
    counts, donor_index, chromosomes = _fixture()
    options = {
        "rank": rank,
        "solver": solver,
        "tol": 1e-10,
        "maxiter": 20,
        "seed": 19,
        "pflog_alpha": 0.2,
    }
    if solver == "arpack":
        options["ncv"] = 5
    options.update(kwargs)
    return state.construct_state_factor(counts, donor_index, chromosomes, **options)


def _projector(values: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(values)
    return basis @ basis.T


def test_state_factor_api_is_public_and_backend_neutral() -> None:
    state = _state_api()

    signature = inspect.signature(state.construct_state_factor)
    assert signature.parameters["solver"].default is inspect.Parameter.empty
    assert signature.parameters["tol"].default is inspect.Parameter.empty
    assert signature.parameters["maxiter"].default is inspect.Parameter.empty
    assert signature.parameters["seed"].default is inspect.Parameter.empty
    assert "scipy" not in repr(state.StateFactorResult).lower()
    result = _construct()
    public_values = [getattr(result, field.name) for field in fields(result)]
    diagnostic_values = [getattr(result.diagnostics, field.name) for field in fields(result.diagnostics)]
    assert not any(isinstance(value, LinearOperator) for value in (*public_values, *diagnostic_values))


@pytest.mark.parametrize("solver", ["propack", "arpack"])
def test_factorization_matches_dense_svd_and_unweighted_scores(solver: str) -> None:
    result = _construct(solver=solver)
    counts, donor_index, chromosomes = _fixture()
    balanced, active = _dense_operator(
        counts,
        donor_index,
        chromosomes,
        alpha=0.2,
        exclude_chromosome=None,
        balance_donors=True,
    )
    unbalanced, _ = _dense_operator(
        counts,
        donor_index,
        chromosomes,
        alpha=0.2,
        exclude_chromosome=None,
        balance_donors=False,
    )
    _, singular_values, vh = np.linalg.svd(balanced, full_matrices=False)
    expected_loadings = vh[:2].T
    expected_factors = unbalanced @ expected_loadings

    np.testing.assert_allclose(result.singular_values, singular_values[:2], rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(_projector(result.loadings), _projector(expected_loadings), atol=2e-9)
    np.testing.assert_allclose(result.factors @ result.factors.T, expected_factors @ expected_factors.T, atol=2e-9)
    np.testing.assert_allclose(result.factors, unbalanced @ result.loadings, atol=2e-12)
    np.testing.assert_array_equal(result.diagnostics.active_gene_indices, active)
    assert np.all(np.diff(result.singular_values) <= 0.0)
    for column in range(result.loadings.shape[1]):
        anchor = int(np.argmax(np.abs(result.loadings[:, column])))
        assert result.loadings[anchor, column] >= 0.0


def test_factorization_preserves_singletons_and_donor_centering() -> None:
    result = _construct()
    _, donor_index, _ = _fixture()

    assert result.factors.shape == (donor_index.size, 2)
    assert np.isfinite(result.factors[donor_index == 2]).all()
    assert np.isfinite(result.factors[donor_index == 3]).all()
    for donor in np.unique(donor_index):
        np.testing.assert_allclose(np.mean(result.factors[donor_index == donor], axis=0), 0.0, atol=2e-15)
    assert result.diagnostics.donor_counts.tolist() == [3, 3, 1, 1]
    assert result.diagnostics.singleton_donor_count == 2


def test_balanced_singular_system_matches_explicit_donor_covariance() -> None:
    result = _construct(rank=4)
    counts, donor_index, chromosomes = _fixture()
    balanced, _ = _dense_operator(
        counts,
        donor_index,
        chromosomes,
        alpha=0.2,
        exclude_chromosome=None,
        balance_donors=True,
    )
    reconstructed_covariance = result.loadings @ np.diag(result.singular_values**2) @ result.loadings.T

    np.testing.assert_allclose(reconstructed_covariance, balanced.T @ balanced, atol=3e-10)
    assert np.all(balanced[donor_index >= 2] == 0.0)


def test_factor_result_is_immutable_read_only_and_deterministic() -> None:
    first = _construct()
    second = _construct()

    np.testing.assert_array_equal(first.factors, second.factors)
    np.testing.assert_array_equal(first.loadings, second.loadings)
    np.testing.assert_array_equal(first.singular_values, second.singular_values)
    assert not first.factors.flags.writeable
    assert not first.loadings.flags.writeable
    assert not first.singular_values.flags.writeable
    assert not first.diagnostics.active_gene_indices.flags.writeable
    assert not first.diagnostics.donor_counts.flags.writeable
    assert first.diagnostics.rank == 2
    assert first.diagnostics.propack_kmax == 7
    assert first.diagnostics.arpack_ncv is None
    with pytest.raises(ValueError, match="read-only"):
        first.factors[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        first.diagnostics.seed = 20


@pytest.mark.parametrize(
    ("shape", "requested_maxiter", "expected_kmax"),
    [
        ((8, 3), 20, 4),
        ((3, 8), 20, 4),
        ((8, 6), 4, 4),
    ],
)
def test_propack_diagnostics_record_effective_kmax(
    shape: tuple[int, int],
    requested_maxiter: int,
    expected_kmax: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jaxqtl.state._factor as factor_module

    def exact(operator, *, k, **kwargs):
        del kwargs
        dense = operator.matmat(np.eye(operator.shape[1], dtype=np.float64))
        _, singular_values, vh = np.linalg.svd(dense, full_matrices=False)
        return None, singular_values[:k], vh[:k]

    monkeypatch.setattr(factor_module.sparse_linalg, "svds", exact)
    n_cells, n_genes = shape
    dense_counts = (np.arange(n_cells * n_genes, dtype=np.int64).reshape(shape) % 5) + 1
    counts = sparse.csr_array(dense_counts)
    donor_index = np.zeros(n_cells, dtype=np.int64)
    chromosomes = np.full(n_genes, "X")

    result = _state_api().construct_state_factor(
        counts,
        donor_index,
        chromosomes,
        rank=1,
        solver="propack",
        tol=1e-10,
        maxiter=requested_maxiter,
        seed=7,
        pflog_alpha=0.2,
        center_within_donor=False,
        balance_donors=False,
    )

    assert result.diagnostics.maxiter == requested_maxiter
    assert result.diagnostics.propack_kmax == expected_kmax


@pytest.mark.parametrize("solver", ["propack", "arpack"])
def test_largest_truncated_rank_is_accepted_but_full_rank_is_rejected(solver: str) -> None:
    rank = 5
    options = {"ncv": None, "center_within_donor": False, "balance_donors": False}
    if solver == "arpack":
        rank = 4
        options.pop("center_within_donor")
        options.pop("balance_donors")
        options["ncv"] = 5
    largest = _construct(solver=solver, rank=rank, **options)
    assert largest.loadings.shape == (6, rank)

    with pytest.raises(ValueError, match="strictly smaller"):
        _construct(solver=solver, rank=6, **options)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"rank": 0}, "rank"),
        ({"solver": "lobpcg"}, "solver"),
        ({"tol": 0.0}, "tol"),
        ({"tol": np.inf}, "tol"),
        ({"maxiter": 0}, "maxiter"),
        ({"seed": -1}, "seed"),
        ({"solver": "propack", "ncv": 4}, "ncv"),
        ({"solver": "arpack", "ncv": None}, "ncv"),
        ({"solver": "arpack", "ncv": 2}, "ncv"),
        ({"solver": "arpack", "ncv": 6}, "ncv"),
    ],
)
def test_invalid_factorization_configuration_is_rejected(updates: dict[str, object], message: str) -> None:
    options: dict[str, object] = {
        "rank": 2,
        "solver": "propack",
        "tol": 1e-8,
        "maxiter": 20,
        "seed": 0,
        "pflog_alpha": 0.2,
    }
    options.update(updates)
    counts, donor_index, chromosomes = _fixture()

    with pytest.raises((TypeError, ValueError), match=message):
        _state_api().construct_state_factor(counts, donor_index, chromosomes, **options)


def test_invalid_rank_is_rejected_before_operator_materialization(monkeypatch) -> None:
    import jaxqtl.state._factor as factor_module

    counts, donor_index, chromosomes = _fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("operator construction must not run for invalid rank")

    monkeypatch.setattr(factor_module, "pflog_operator", forbidden)
    with pytest.raises(ValueError, match="strictly smaller"):
        _state_api().construct_state_factor(
            counts,
            donor_index,
            chromosomes,
            rank=6,
            solver="propack",
            tol=1e-9,
            maxiter=20,
            seed=4,
            pflog_alpha=0.2,
        )


def test_loco_matches_explicit_exclusion_and_has_no_held_out_leakage() -> None:
    state = _state_api()
    counts, donor_index, chromosomes = _fixture()
    common = {"rank": 2, "solver": "propack", "tol": 1e-10, "maxiter": 20, "seed": 11}
    loco = state.construct_state_factor(
        counts,
        donor_index,
        chromosomes,
        exclude_chromosome="1",
        pflog_alpha="auto",
        **common,
    )
    retained = chromosomes != "1"
    explicit = state.construct_state_factor(
        sparse.csr_array(counts[:, retained]),
        donor_index,
        chromosomes[retained],
        pflog_alpha="auto",
        **common,
    )
    perturbed = np.asarray(counts.toarray()).copy()
    held_out = np.flatnonzero(~retained)
    perturbed[:, held_out] = perturbed[:, held_out] * 1000 + 17
    no_leak = state.construct_state_factor(
        sparse.csr_array(perturbed),
        donor_index,
        chromosomes,
        exclude_chromosome="1",
        pflog_alpha="auto",
        **common,
    )

    assert loco.diagnostics.alpha == pytest.approx(explicit.diagnostics.alpha, rel=2e-15)
    np.testing.assert_allclose(_projector(loco.factors), _projector(explicit.factors), atol=2e-9)
    np.testing.assert_allclose(loco.factors, no_leak.factors, atol=2e-11)
    np.testing.assert_allclose(loco.loadings, no_leak.loadings, atol=2e-11)
    np.testing.assert_allclose(loco.singular_values, no_leak.singular_values, atol=2e-11)


def test_transform_filtering_counts_are_distinct_from_pflog_fit_counts() -> None:
    counts = sparse.csr_array(
        np.asarray(
            [
                [1, 0, 1],
                [1, 1, 0],
                [1, 3, 2],
                [1, 6, 5],
            ],
            dtype=np.int64,
        )
    )
    donor_index = np.asarray([0, 0, 1, 1], dtype=np.int64)
    chromosomes = np.asarray(["1", "2", "X"])

    result = _state_api().construct_state_factor(
        counts,
        donor_index,
        chromosomes,
        rank=1,
        solver="propack",
        tol=1e-9,
        maxiter=20,
        seed=3,
        exclude_chromosome="1",
        pflog_alpha=0.2,
    )

    assert hasattr(result.diagnostics, "input_gene_count")
    assert hasattr(result.diagnostics, "transform_excluded_gene_count")
    assert result.diagnostics.input_gene_count == 3
    assert result.diagnostics.active_gene_indices.size == 2
    assert result.diagnostics.transform_excluded_gene_count == 1
    assert result.diagnostics.alpha_excluded_gene_count == 0


def test_default_loco_iterator_is_streaming_ordered_and_accepts_absent_autosomes() -> None:
    state = _state_api()
    counts, donor_index, chromosomes = _fixture()
    iterator = state.iter_loco_state_factors(
        counts,
        donor_index,
        chromosomes,
        rank=1,
        solver="propack",
        tol=1e-9,
        maxiter=20,
        seed=3,
    )
    assert inspect.isgenerator(iterator)
    results = cast("list[StateFactorResult]", list(iterator))

    assert [result.diagnostics.excluded_chromosome for result in results] == [str(i) for i in range(1, 23)]
    absent = results[3]
    assert absent.diagnostics.alpha_excluded_gene_count == 0
    assert absent.diagnostics.alpha_excluded_numerator == 0.0
    assert absent.diagnostics.alpha_excluded_denominator == 0.0
    no_exclusion = state.construct_state_factor(
        counts,
        donor_index,
        chromosomes,
        rank=1,
        solver="propack",
        tol=1e-9,
        maxiter=20,
        seed=3,
        pflog_alpha="auto",
    )
    np.testing.assert_allclose(absent.factors, no_exclusion.factors, atol=2e-10)


def test_loco_iterator_preserves_requested_order_and_computes_statistics_once(monkeypatch) -> None:
    import jaxqtl.state._factor as factor_module

    state = _state_api()
    counts, donor_index, chromosomes = _fixture()
    original = factor_module.compute_pflog_statistics
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(factor_module, "compute_pflog_statistics", counted)
    results = list(
        state.iter_loco_state_factors(
            counts,
            donor_index,
            chromosomes,
            chromosomes=("3", "1", "4"),
            rank=1,
            solver="propack",
            tol=1e-9,
            maxiter=20,
            seed=3,
        )
    )

    assert calls == 1
    assert [result.diagnostics.excluded_chromosome for result in results] == ["3", "1", "4"]


def test_dense_count_conversion_is_forbidden_and_transformed_storage_is_single(monkeypatch) -> None:
    import jaxqtl.state._operator as operator_module

    state = _state_api()
    counts, donor_index, chromosomes = _fixture()
    guarded = _DenseConversionForbiddenCSR(counts)
    original_transform = operator_module._transformed_sparse_counts
    transform_calls = 0

    def counted_transform(*args, **kwargs):
        nonlocal transform_calls
        transform_calls += 1
        return original_transform(*args, **kwargs)

    monkeypatch.setattr(operator_module, "_transformed_sparse_counts", counted_transform)

    result = state.construct_state_factor(
        guarded,
        donor_index,
        chromosomes,
        rank=2,
        solver="propack",
        tol=1e-9,
        maxiter=20,
        seed=4,
        pflog_alpha=0.2,
    )

    assert transform_calls == 1
    assert result.diagnostics.transformed_shape == counts.shape
    assert result.diagnostics.transformed_nnz <= counts.nnz
    assert result.diagnostics.factors_shape == (counts.shape[0], 2)
    assert result.diagnostics.loadings_shape == (counts.shape[1], 2)


def test_operator_block_actions_never_allocate_a_transformed_count_shape(monkeypatch) -> None:
    from jaxqtl.state._operator import PFLogOperator

    observed_shapes: list[tuple[int, int]] = []
    original_matmat = PFLogOperator.matmat
    original_rmatmat = PFLogOperator.rmatmat

    def recording_matmat(self, matrix):
        result = original_matmat(self, matrix)
        observed_shapes.append(result.shape)
        return result

    def recording_rmatmat(self, matrix):
        result = original_rmatmat(self, matrix)
        observed_shapes.append(result.shape)
        return result

    monkeypatch.setattr(PFLogOperator, "matmat", recording_matmat)
    monkeypatch.setattr(PFLogOperator, "rmatmat", recording_rmatmat)
    result = _construct()

    assert observed_shapes
    assert result.diagnostics.transformed_shape not in observed_shapes
    assert all(shape[1] <= result.loadings.shape[1] for shape in observed_shapes)


def test_arpack_nonconvergence_never_returns_partial_triplets(monkeypatch) -> None:
    import jaxqtl.state._factor as factor_module

    def fail(*args, **kwargs):
        raise ArpackNoConvergence("forced ARPACK failure", np.asarray([1.0]), np.ones((6, 1)))

    monkeypatch.setattr(factor_module.sparse_linalg, "svds", fail)
    with pytest.raises(RuntimeError, match="ARPACK.*converge"):
        _construct(solver="arpack")


def test_propack_linalg_failure_is_actionable(monkeypatch) -> None:
    import jaxqtl.state._factor as factor_module

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("forced PROPACK failure")

    monkeypatch.setattr(factor_module.sparse_linalg, "svds", fail)
    with pytest.raises(RuntimeError, match="PROPACK.*failed"):
        _construct(solver="propack")


@pytest.mark.parametrize(
    ("singular_values", "message"),
    [
        (np.asarray([np.nan, 1.0]), "singular"),
        (np.asarray([1.0, 0.0]), "floor"),
    ],
)
def test_nonfinite_or_near_zero_requested_triplets_are_rejected(monkeypatch, singular_values, message) -> None:
    import jaxqtl.state._factor as factor_module

    def invalid(_operator, *, k, **kwargs):
        del kwargs
        return None, singular_values[:k], np.eye(_operator.shape[1], dtype=np.float64)[:k]

    monkeypatch.setattr(factor_module.sparse_linalg, "svds", invalid)
    with pytest.raises(RuntimeError, match=message):
        _construct(rank=2)


def test_incomplete_solver_output_is_rejected(monkeypatch) -> None:
    import jaxqtl.state._factor as factor_module

    def incomplete(_operator, *, k, **kwargs):
        del kwargs
        return None, np.ones(k - 1), np.eye(_operator.shape[1], dtype=np.float64)[: k - 1]

    monkeypatch.setattr(factor_module.sparse_linalg, "svds", incomplete)
    with pytest.raises(RuntimeError, match="incomplete"):
        _construct()


def test_residual_validation_rejects_inaccurate_triplets(monkeypatch) -> None:
    import jaxqtl.state._factor as factor_module

    def inaccurate(operator, *, k, **kwargs):
        del kwargs
        dense = operator.matmat(np.eye(operator.shape[1], dtype=np.float64))
        _, singular_values, vh = np.linalg.svd(dense, full_matrices=False)
        singular_values = singular_values[:k].copy()
        singular_values[0] *= 1.01
        return None, singular_values, vh[:k]

    monkeypatch.setattr(factor_module.sparse_linalg, "svds", inaccurate)
    with pytest.raises(RuntimeError, match="residual"):
        _construct(tol=1e-10)


def test_scale_aware_triplet_floor_accepts_below_and_rejects_above_boundary(monkeypatch) -> None:
    import jaxqtl.state._factor as factor_module

    def exact(operator, *, k, **kwargs):
        del kwargs
        dense = operator.matmat(np.eye(operator.shape[1], dtype=np.float64))
        _, singular_values, vh = np.linalg.svd(dense, full_matrices=False)
        return None, singular_values[:k], vh[:k]

    monkeypatch.setattr(factor_module.sparse_linalg, "svds", exact)
    counts, donor_index, chromosomes = _fixture()
    balanced, _ = _dense_operator(
        counts,
        donor_index,
        chromosomes,
        alpha=0.2,
        exclude_chromosome=None,
        balance_donors=True,
    )
    singular_values = np.linalg.svd(balanced, compute_uv=False)
    boundary = float(singular_values[1] / singular_values[0])

    accepted = _construct(tol=np.nextafter(boundary, 0.0))
    assert accepted.singular_values[-1] > accepted.diagnostics.sigma_floor
    with pytest.raises(RuntimeError, match="floor"):
        _construct(tol=np.nextafter(boundary, np.inf))


def test_residual_limit_accepts_below_and_rejects_above_boundary(monkeypatch) -> None:
    import jaxqtl.state._factor as factor_module

    target = 0.0

    def controlled(operator, *, k, **kwargs):
        del kwargs
        dense = operator.matmat(np.eye(operator.shape[1], dtype=np.float64))
        _, singular_values, vh = np.linalg.svd(dense, full_matrices=False)
        adjusted = singular_values[:k] / np.sqrt(1.0 - target)
        return None, adjusted, vh[:k]

    monkeypatch.setattr(factor_module.sparse_linalg, "svds", controlled)
    tol = 1e-4
    limit = 10.0 * tol
    target = limit * (1.0 - 1e-6)
    accepted = _construct(rank=1, tol=tol)
    assert accepted.diagnostics.max_adjoint_residual <= accepted.diagnostics.residual_limit

    target = limit * (1.0 + 1e-6)
    with pytest.raises(RuntimeError, match="residual"):
        _construct(rank=1, tol=tol)


def test_orthogonality_limit_accepts_below_and_rejects_above_boundary(monkeypatch) -> None:
    import jaxqtl.state._factor as factor_module

    correlation = 0.0
    identity_operator = _IdentityDomainOperator()

    def operator_factory(*args, **kwargs):
        del args, kwargs
        return identity_operator

    def same_view(operator, *, balance_donors):
        del balance_donors
        return operator

    def controlled(_operator, *, k, **kwargs):
        del kwargs
        vh = np.zeros((k, _operator.shape[1]), dtype=np.float64)
        vh[0, 0] = 1.0
        vh[1, 0] = correlation
        vh[1, 1] = np.sqrt(1.0 - correlation**2)
        return None, np.ones(k, dtype=np.float64), vh

    monkeypatch.setattr(factor_module, "pflog_operator", operator_factory)
    monkeypatch.setattr(factor_module, "_operator_with_balancing", same_view)
    monkeypatch.setattr(factor_module.sparse_linalg, "svds", controlled)
    tol = 1e-4
    limit = 10.0 * tol
    correlation = limit * (1.0 - 1e-6)
    accepted = _construct(rank=2, tol=tol)
    assert accepted.diagnostics.loading_orthogonality_error <= accepted.diagnostics.residual_limit

    correlation = limit * (1.0 + 1e-6)
    with pytest.raises(RuntimeError, match="orthogonality"):
        _construct(rank=2, tol=tol)


def test_repeated_singular_values_are_compared_by_subspace(monkeypatch) -> None:
    import jaxqtl.state._factor as factor_module

    identity_operator = _IdentityDomainOperator()
    angle = 0.0

    def operator_factory(*args, **kwargs):
        del args, kwargs
        return identity_operator

    def same_view(operator, *, balance_donors):
        del balance_donors
        return operator

    def repeated(_operator, *, k, **kwargs):
        del kwargs
        cosine = np.cos(angle)
        sine = np.sin(angle)
        vh = np.zeros((k, _operator.shape[1]), dtype=np.float64)
        vh[0, :2] = (cosine, sine)
        vh[1, :2] = (-sine, cosine)
        return None, np.ones(k, dtype=np.float64), vh

    monkeypatch.setattr(factor_module, "pflog_operator", operator_factory)
    monkeypatch.setattr(factor_module, "_operator_with_balancing", same_view)
    monkeypatch.setattr(factor_module.sparse_linalg, "svds", repeated)
    first = _construct(rank=2)
    angle = np.pi / 5.0
    rotated = _construct(rank=2)

    np.testing.assert_allclose(_projector(first.loadings), _projector(rotated.loadings), atol=2e-15)
    np.testing.assert_allclose(_projector(first.factors), _projector(rotated.factors), atol=2e-15)
    assert not np.allclose(first.loadings, rotated.loadings)
