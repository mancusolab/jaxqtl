# pattern: Functional Core

from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp

from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from ..distribution import (
    ExponentialFamily,
    Gaussian,
    IdentityLink,
    NegativeBinomial,
    Poisson,
    t_cdf,
)
from ._optimize import irls, lstsq
from ._solve import AbstractLinearSolve, CholeskySolve
from ._stderr import AbstractVarianceEstimator, FisherInfoError


class ModelResult(NamedTuple):
    r"""Container for fitted model outputs.

    This stores coefficient estimates and derived quantities returned by [`jaxqtl.infer.LinearModel.fit`][] and
    [`jaxqtl.infer.GeneralizedLinearModel.fit`][]. These outputs are consumed by downstream hypothesis tests
    and mapping routines. ``eta`` is the complete linear predictor, including any supplied offset, and ``mu`` is
    the corresponding fitted mean. ``resid`` is the working residual used by score tests; for the Gaussian
    identity-link model it is ``y - mu``.
    """

    beta: Array
    se: Array
    z: Array
    p: Array
    eta: Array
    mu: Array
    glm_wt: Array
    link_prime: Array
    num_iters: Array
    converged: Array
    resid_covar: Array  # covariance used to compute `se`
    resid: Array  # working residual used by score tests
    disp: Array  # dispersion parameter


class _AbstractInit(eqx.Module):
    """Annoying, but let's split out how to init the glm here. Most of the time this isn't needed, but NegBin
    really benefits from first initializing a Poisson family to calculate/estimate dispersion ahead of time.
    """

    family: eqx.AbstractVar[ExponentialFamily]
    solver: eqx.AbstractVar[AbstractLinearSolve]

    def __call__(
        self,
        X: Array,
        y: Array,
        offset: Array,
        max_iter: int = 100,
        tol: float = 1e-3,
        step_size: float = 1e-2,
    ):
        return self.init(X, y, offset, max_iter, tol=tol, step_size=step_size)

    @abstractmethod
    def init(
        self,
        X: Array,
        y: Array,
        offset: Array,
        max_iter: int = 100,
        tol: float = 1e-3,
        step_size: float = 1e-2,
    ):
        pass


class _NBInit(_AbstractInit):
    family: ExponentialFamily
    solver: AbstractLinearSolve

    def __post_init__(self):
        if not isinstance(self.family, NegativeBinomial):
            raise ValueError("_NBInit only supports NegativeBinomial")
        return

    def init(
        self,
        X: Array,
        y: Array,
        offset: Array,
        max_iter: int = 100,
        tol: float = 1e-3,
        step_size: float = 1e-2,
    ):
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        offset = jnp.asarray(offset)
        n, p = X.shape

        jaxqtl_pois = GeneralizedLinearModel(
            family=Poisson(), solver=self.solver, max_iter=max_iter, tol=tol, step_size=step_size
        )
        glm_state_pois = jaxqtl_pois.fit(X, y, offset)

        # fit covariate-only model (null)
        disp_init = n / jnp.sum((y / self.family.glink.inverse(glm_state_pois.eta) - 1) ** 2)
        eta = glm_state_pois.eta
        disp = self.family.estimate_dispersion(X, y, eta, disp=1.0 / disp_init, max_iter=max_iter)

        # convert disp to 0.1 if bad initialization
        disp = jnp.nan_to_num(disp, nan=0.1)

        return eta, disp


class _SimpleInit(_AbstractInit):
    family: ExponentialFamily
    solver: AbstractLinearSolve

    def init(
        self,
        X: Array,
        y: Array,
        offset: Array,
        max_iter: int = 100,
        tol: float = 1e-3,
        step_size: float = 1e-2,
    ):
        y = jnp.asarray(y)
        init_val = self.family.init_eta(y)
        return init_val, jnp.array(1.0)


class AbstractLinearModel(eqx.Module):
    r"""Abstract base class for linear and generalized linear models."""

    family: eqx.AbstractVar[ExponentialFamily]
    solver: eqx.AbstractVar[AbstractLinearSolve]

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike = 0.0,
        std_err: AbstractVarianceEstimator = FisherInfoError(),
    ) -> ModelResult:
        r"""Fit a model and return a summary state.

        **Arguments:**

        - `X`: Design matrix with shape `(n, p)`.
        - `y`: Response vector with shape `(n,)`.
        - `offset`: Offset broadcastable to `y` (either scalar or `(n,)`).
        - `std_err`: Coefficient covariance estimator implementing [`jaxqtl.infer.AbstractVarianceEstimator`][].

        **Returns:**

        A [`jaxqtl.infer.ModelResult`][] containing fitted coefficients, standard errors, and auxiliary quantities.
        """
        pass


class LinearModel(AbstractLinearModel):
    r"""Gaussian linear regression with a fast least-squares implementation.

    This model requires an identity link and avoids the full IRLS loop.

    **Raises:**

    - `ValueError`: If `family` is not Gaussian with an identity link.
    """

    family: ExponentialFamily = Gaussian()
    solver: AbstractLinearSolve = CholeskySolve()

    def __check_init__(self) -> None:
        if not isinstance(self.family, Gaussian) or not isinstance(self.family.glink, IdentityLink):
            raise ValueError("LinearModel only supports Gaussian family with IdentityLink")

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike = 0.0,
        std_err: AbstractVarianceEstimator = FisherInfoError(),
        *,
        df_resid: int | None = None,
    ) -> ModelResult:
        r"""Fit a Gaussian linear model and return a summary state.

        **Arguments:**

        - `X`: Design matrix with shape `(n, p)`.
        - `y`: Response vector with shape `(n,)`.
        - `offset`: Offset broadcastable to `y` (either scalar or `(n,)`).
        - `std_err`: Coefficient covariance estimator implementing [`jaxqtl.infer.AbstractVarianceEstimator`][].
        - `df_resid`: Residual degrees of freedom used for both the residual-dispersion denominator and the Student's t
          reference distribution. Defaults to `n - p`; callers that fit a residualized submodel may provide the
          degrees of freedom from the corresponding full model.

        **Returns:**

        A [`jaxqtl.infer.ModelResult`][] containing fitted coefficients, standard errors, and auxiliary quantities.

        **Raises:**

        - `ValueError`: If `df_resid` is not positive. The default also raises when the design has no residual degrees
          of freedom.
        """
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        offset = jnp.asarray(offset)
        df = X.shape[0] - X.shape[1] if df_resid is None else df_resid
        if df <= 0:
            raise ValueError(f"LinearModel requires positive residual degrees of freedom; received {df}.")
        beta, n_iter, converged, _ = lstsq(X, y - offset, self.solver)

        eta = X @ beta + offset
        # The Gaussian model enforces the identity link, so its fitted mean is
        # the complete linear predictor, including the fixed offset.
        mu = eta
        resid = y - mu
        disp = jnp.sum(resid**2) / df

        weight = 1.0 / disp

        resid_covar = std_err(self.family, X, y, eta, mu, weight, disp)
        beta_se = jnp.sqrt(jnp.diag(resid_covar))

        stat = beta / beta_se
        pval_wald = 2 * t_cdf(-jnp.abs(stat), df)

        return ModelResult(
            beta,
            beta_se,
            stat,
            pval_wald,
            eta,
            mu,
            weight,
            jnp.ones_like(weight),
            n_iter,
            converged,
            resid_covar,
            resid,
            disp,
        )


class GeneralizedLinearModel(AbstractLinearModel):
    r"""Generalized linear model (GLM) fitted via IRLS.

    This class wraps a family (distribution + link) and a linear solver, and fits coefficients using
    iteratively reweighted least squares (IRLS).
    """

    family: ExponentialFamily = Gaussian()
    solver: AbstractLinearSolve = CholeskySolve()

    max_iter: int = 1000
    tol: float = 1e-3
    step_size: float = 1.0

    _init: _AbstractInit = eqx.field(init=False)

    def __post_init__(self):
        if isinstance(self.family, NegativeBinomial):
            # this is kind of architectually annoying. NB family shouldn't know about the GLM
            # and we need to init NB using Poisson GLM, so this is our hack/workaround for now.
            self._init = _NBInit(self.family, self.solver)
        else:
            self._init = _SimpleInit(self.family, self.solver)

        return

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike = 0.0,
        std_err: AbstractVarianceEstimator = FisherInfoError(),
    ) -> ModelResult:
        r"""Fit a GLM with IRLS and return a summary state.

        **Arguments:**

        - `X`: Design matrix with shape `(n, p)`.
        - `y`: Response vector with shape `(n,)`.
        - `offset`: Offset broadcastable to `y` (either scalar or `(n,)`).
        - `std_err`: Coefficient covariance estimator implementing [`jaxqtl.infer.AbstractVarianceEstimator`][].

        **Returns:**

        A [`jaxqtl.infer.ModelResult`][] containing fitted coefficients, standard errors, and auxiliary quantities.
        """
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        offset = jnp.asarray(offset)

        # initialize eta and alpha
        init, disp_init = self._init(X, y, offset, self.max_iter, self.tol, self.step_size)
        beta, n_iter, converged, disp = irls(
            X, y, offset, init, self.family, self.solver, self.max_iter, self.tol, self.step_size, disp_init
        )
        eta = X @ beta + offset
        mu, link_prime, weight = self.family.calc_weight(eta, disp)
        resid = (y - mu) * link_prime  # note: this is the working resid

        resid_covar = std_err(self.family, X, y, eta, mu, weight, disp)
        beta_se = jnp.sqrt(jnp.diag(resid_covar))
        stat = beta / beta_se
        pval_wald = 2 * norm.sf(jnp.abs(stat))

        return ModelResult(
            beta,
            beta_se,
            stat,
            pval_wald,
            eta,
            mu,
            weight,
            link_prime,
            n_iter,
            converged,
            resid_covar,
            resid,
            disp,
        )
