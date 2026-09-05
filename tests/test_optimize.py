# pattern: Functional Core

from typing import ClassVar

import numpy as np
import pytest

import equinox as eqx
import jax
import jax.numpy as jnp

from jax import config, random

from jaxqtl.distribution._expfam import ExponentialFamily, NegativeBinomial, Poisson
from jaxqtl.distribution._links import AbstractLink, IdentityLink
from jaxqtl.infer import irls
from jaxqtl.infer._optimize import infer_beta_params, lstsq
from jaxqtl.infer._solve import AbstractLinearSolve, CholeskySolve, QRSolve


config.update("jax_enable_x64", True)


class _QuadraticTrialFamily(ExponentialFamily):
    glink: AbstractLink
    optimum: float
    nonfinite_above: float
    _valid_links: ClassVar[list[type[AbstractLink]]] = [IdentityLink]

    def __init__(self, optimum: float, nonfinite_above: float = float("inf")):
        self.glink = IdentityLink()
        self.optimum = optimum
        self.nonfinite_above = nonfinite_above

    def scale(self, X, y, mu):
        return jnp.asarray(1.0)

    def negloglikelihood(self, X, y, eta, disp):
        objective = jnp.sum(jnp.square(jnp.asarray(eta) - self.optimum))
        return jnp.where(jnp.any(jnp.asarray(eta) > self.nonfinite_above), jnp.nan, objective)

    def variance(self, mu, disp=1.0):
        return jnp.ones_like(jnp.asarray(mu))

    def sample(self, key, eta, disp=1.0):
        return jnp.asarray(eta)


def _run_single_iteration(family, y, disp_init=1.0):
    return irls(
        jnp.ones((1, 1)),
        jnp.asarray([y]),
        jnp.asarray(0.0),
        jnp.asarray([0.0]),
        family,
        CholeskySolve(),
        max_iter=1,
        tol=0.0,
        step_size=1.0,
        disp_init=jnp.asarray(disp_init),
    )


@pytest.mark.parametrize("step_size", (0.0, -1.0, float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("parameter", ("step_size", "gtol"))
def test_irls_rejects_invalid_solver_control_at_public_boundary(step_size, parameter):
    with pytest.raises(ValueError, match=f"{parameter} must be finite and greater than 0"):
        irls(
            jnp.ones((1, 1)),
            jnp.asarray([1.0]),
            jnp.asarray(0.0),
            jnp.asarray([0.0]),
            _QuadraticTrialFamily(optimum=0.0),
            CholeskySolve(),
            max_iter=1,
            disp_init=jnp.asarray(1.0),
            **{parameter: step_size},
        )


def test_irls_public_wrapper_supports_outer_filter_jit():
    state = eqx.filter_jit(irls)(
        jnp.ones((1, 1)),
        jnp.asarray([2.0]),
        jnp.asarray(0.0),
        jnp.asarray([0.0]),
        _QuadraticTrialFamily(optimum=0.75),
        CholeskySolve(),
        max_iter=1,
        tol=0.0,
        step_size=1.0,
        disp_init=jnp.asarray(1.0),
    )

    np.testing.assert_allclose(np.asarray(state.beta), [1.0])


def test_irls_rejects_objective_increase_then_accepts_halved_step():
    state = _run_single_iteration(_QuadraticTrialFamily(optimum=0.75), y=2.0)

    np.testing.assert_allclose(np.asarray(state.beta), [1.0])


def test_irls_rejects_nonfinite_candidate_then_accepts_halved_step():
    state = _run_single_iteration(_QuadraticTrialFamily(optimum=0.75, nonfinite_above=1.5), y=2.0)

    np.testing.assert_allclose(np.asarray(state.beta), [1.0])


def test_irls_exhaustion_preserves_prior_finite_state_and_reports_failure():
    state = _run_single_iteration(_QuadraticTrialFamily(optimum=0.0), y=1.0, disp_init=2.0)

    np.testing.assert_allclose(np.asarray(state.beta), [0.0])
    np.testing.assert_allclose(np.asarray(state.disp), 2.0)
    assert np.all(np.isfinite(np.asarray(state.beta)))
    assert np.isfinite(np.asarray(state.disp))
    assert not bool(state.converged)


def test_irls_small_likelihood_change_does_not_hide_large_beta_gradient():
    state = irls(
        jnp.ones((4, 1)),
        jnp.full(4, 2.0),
        jnp.asarray(0.0),
        jnp.zeros(4),
        Poisson(),
        CholeskySolve(),
        max_iter=2,
        step_size=1e-6,
        disp_init=1.0,
    )

    assert state.num_iters == 2
    assert not bool(state.converged)


@pytest.mark.parametrize(("counts", "converged"), [([0, 1, 1, 2], True), ([0, 0, 0, 4], False)])
def test_irls_lower_dispersion_bound_requires_correct_gradient_sign(counts, converged):
    # Both have mean one. A sub-roundoff step leaves the parameters unchanged;
    # only the underdispersed sample has a constrained optimum at the floor.
    state = irls(
        jnp.ones((4, 1)),
        jnp.asarray(counts, dtype=float),
        jnp.asarray(0.0),
        jnp.zeros(4),
        NegativeBinomial(),
        CholeskySolve(),
        max_iter=2,
        step_size=1e-30,
        disp_init=1e-9,
    )

    assert bool(state.converged) == converged


def test_irls_backtracking_uses_one_weighted_solve():
    calls = []

    class CountingSolve(AbstractLinearSolve):
        def wgt_lstsq(self, X, r, weights):
            jax.debug.callback(lambda: calls.append(1))
            return CholeskySolve().wgt_lstsq(X, r, weights)

        def lstsq(self, X, r):
            return CholeskySolve().lstsq(X, r)

    state = irls(
        jnp.ones((1, 1)),
        jnp.asarray([16.0]),
        jnp.asarray(0.0),
        jnp.zeros(1),
        _QuadraticTrialFamily(optimum=0.75),
        CountingSolve(),
        max_iter=1,
        disp_init=1.0,
    )
    jax.block_until_ready(state)
    jax.effects_barrier()

    np.testing.assert_allclose(np.asarray(state.beta), [1.0])
    assert len(calls) == 1


def test_irls_gradient_check_is_invariant_to_column_units_under_vmap():
    X = jnp.column_stack((jnp.ones(6), jnp.asarray([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0])))
    y = jnp.asarray([0.0, 1.0, 1.0, 2.0, 2.0, 4.0])
    scales = jnp.asarray([1e-4, 1.0, 1e4])

    @eqx.filter_jit
    @eqx.filter_vmap
    def fit(scale):
        design = X * jnp.asarray([1.0, scale])
        return irls(
            design, y, jnp.asarray(0.0), jnp.zeros(6), Poisson(), CholeskySolve(), tol=1.0, gtol=1e-6, disp_init=1.0
        )

    states = fit(scales)
    single = irls(X, y, jnp.asarray(0.0), jnp.zeros(6), Poisson(), CholeskySolve(), tol=1.0, gtol=1e-6, disp_init=1.0)

    assert jnp.all(states.converged)
    np.testing.assert_array_equal(states.num_iters, jnp.full(3, single.num_iters))
    np.testing.assert_allclose(states.beta[:, 0], single.beta[0], atol=1e-10)
    np.testing.assert_allclose(states.beta[:, 1] * scales, single.beta[1], atol=1e-10)
    # Independent Poisson NLL gradient at the returned coefficients.
    gradient = X.T @ (jnp.exp(X @ single.beta) - y)
    error = jnp.max(jnp.abs(gradient) / (y.size * jnp.sqrt(jnp.mean(X**2, axis=0))))
    assert error <= 1e-6


@pytest.mark.parametrize("solver", [QRSolve(), CholeskySolve()])
def test_lstsq_matches_normal_equations(solver):
    key = random.PRNGKey(0)
    n, p = 60, 5

    key, key_x, key_b = random.split(key, 3)
    X_raw = random.normal(key_x, shape=(n, p))
    X, _ = jnp.linalg.qr(X_raw)
    beta_true = random.normal(key_b, shape=(p,))
    y = X @ beta_true

    state = lstsq(X, y, solver)

    beta_expected = jnp.linalg.solve(X.T @ X, X.T @ y)
    np.testing.assert_allclose(np.asarray(state.beta), np.asarray(beta_expected), rtol=1e-6, atol=1e-6)
    assert state.num_iters == 1
    assert bool(state.converged)
    np.testing.assert_allclose(np.asarray(state.disp), 1.0)


@pytest.mark.parametrize("solver", [QRSolve(), CholeskySolve()])
def test_irls_poisson_matches_statsmodels(solver):
    statsmodels = pytest.importorskip("statsmodels.api")

    key = random.PRNGKey(1)
    n, p = 800, 4

    key, key_x, key_b, key_off, key_y = random.split(key, 5)
    X = random.normal(key_x, shape=(n, p))
    X = X.at[:, 0].set(1.0)

    beta_true = jnp.array([0.3, -0.2, 0.15, 0.05]) + 0.1 * random.normal(key_b, shape=(p,))
    offset = 0.1 * random.normal(key_off, shape=(n,))

    eta_true = X @ beta_true + offset
    y = random.poisson(key_y, lam=jnp.exp(eta_true))

    family = Poisson()
    init_eta = family.init_eta(y)

    state = irls(
        X,
        y,
        offset,
        init_eta,
        family,
        solver,
        max_iter=200,
        tol=1e-6,
        step_size=1.0,
        disp_init=jnp.asarray(1.0),
    )

    assert bool(state.converged)

    sm_model = statsmodels.GLM(
        np.asarray(y),
        np.asarray(X),
        family=statsmodels.families.Poisson(),
        offset=np.asarray(offset),
    )
    sm_res = sm_model.fit()

    np.testing.assert_allclose(np.asarray(state.beta), np.asarray(sm_res.params), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("true_k", "true_n"),
    [
        (1.5, 1000.0),
        (2.0, 5.0),
        (0.8, 3.0),
    ],
)
def test_infer_beta_params_improves_loglik(true_k, true_n):
    from jax.scipy import stats as jaxstats

    key = random.PRNGKey(2)
    sample_n = 800
    p_perm = random.beta(key, a=true_k, b=true_n, shape=(sample_n,))

    p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
    p_var = jnp.maximum(p_var, jnp.finfo(float).eps)
    shape_term = jnp.maximum(p_mean * (1 - p_mean) / p_var - 1, 1e-3)
    k_init = p_mean * shape_term
    n_init = k_init * (1 / p_mean - 1)
    init = jnp.array([k_init, n_init])

    res = infer_beta_params(p_perm, init, step_size=1.0)

    assert jnp.isfinite(res.k)
    assert jnp.isfinite(res.n)
    assert res.k > 0
    assert res.n > 0

    init_lik = jnp.sum(jaxstats.beta.logpdf(p_perm, init[0], init[1]))
    final_lik = jnp.sum(jaxstats.beta.logpdf(p_perm, res.k, res.n))
    assert final_lik > init_lik
