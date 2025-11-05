from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx

from jax import Array, numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import ArrayLike

from ..families.distribution import (
    ExponentialFamily,
    Gaussian,
    NegativeBinomial,
    Poisson,
)
from ..families.utils import t_cdf
from .optimize import irls, lstsq
from .solve import CholeskySolve, LinearSolve
from .stderr import ErrVarEstimation, FisherInfoError


class GLMState(NamedTuple):
    beta: Array
    se: Array
    z: Array
    p: Array
    eta: Array
    mu: Array
    glm_wt: Array
    num_iters: Array
    converged: Array
    infor_inv: Array  # for score test
    resid: Array  # for score test, not the working resid!
    alpha: Array  # dispersion parameter in NB model


class _AbstractInit(eqx.Module):
    """Annoying, but lets split out how to FIT the glm here. This is an optimization step to really determine at
    runtime whether we're fitting IRLS for the GLM or a faster/simpler fit for a LM
    """

    family: eqx.AbstractVar[ExponentialFamily]
    solver: eqx.AbstractVar[LinearSolve]
    std_err: eqx.AbstractVar[ErrVarEstimation]

    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike,
        max_iter: int = 100,
        tol: float = 1e-3,
        step_size: float = 1e-2,
    ):
        return self.init(X, y, offset_eta, max_iter, tol=tol, step_size=step_size)

        pass
    @abstractmethod
    def init(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike,
        max_iter: int = 100,
        tol: float = 1e-3,
        step_size: float = 1e-2,
    ):
        pass


class _NBInit(_AbstractInit):
    family: ExponentialFamily
    solver: LinearSolve
    std_err: ErrVarEstimation

    def __post_init__(self):
        if not isinstance(self.family, NegativeBinomial):
            raise ValueError("_NBInit only supports NegativeBinomial")
        return

    def init(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike,
        max_iter: int = 100,
        tol: float = 1e-3,
        step_size: float = 1e-2,
    ):
        n, p = X.shape
        init_val = self.family.init_eta(y)

        jaxqtl_pois = GLM(family=Poisson(), solver=self.solver, std_err=self.std_err)
        glm_state_pois = jaxqtl_pois.fit(X, y, offset=offset_eta)

        # fit covariate-only model (null)
        alpha_init = n / jnp.sum((y / self.family.glink.inverse(glm_state_pois.eta) - 1) ** 2)
        eta = glm_state_pois.eta
        disp = self.family.estimate_dispersion(X, y, eta, alpha=1.0 / alpha_init, max_iter=max_iter)

        # convert disp to 0.1 if bad initialization
        disp = jnp.nan_to_num(disp, nan=0.1)

        return eta, disp


class _SimpleInit(_AbstractInit):
    family: ExponentialFamily
    solver: LinearSolve
    std_err: ErrVarEstimation

    def init(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike,
        max_iter: int = 100,
        tol: float = 1e-3,
        step_size: float = 1e-2,
    ):
        init_val = self.family.init_eta(y)
        return init_val, jnp.array(0.0)


class AbstractLinearModel(eqx.Module):
    family: eqx.AbstractVar[ExponentialFamily]
    solver: eqx.AbstractVar[LinearSolve]
    std_err: eqx.AbstractVar[ErrVarEstimation]

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike = 0.0,
    ):
        pass


class LinearModel(AbstractLinearModel):
    family: ExponentialFamily = Gaussian()
    solver: LinearSolve = CholeskySolve()
    std_err: ErrVarEstimation = FisherInfoError()

    def __post_init__(self):
        if not isinstance(self.family, Gaussian):
            raise ValueError("LinearModel only supports Gaussian family")
        return

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike = 0.0,
    ):
        beta, n_iter, converged, alpha = lstsq(X, y - offset, self.solver)

        mu = X @ beta
        eta = mu
        resid = y - mu - offset  # note: this is the working resid

        phi = self.family.scale(X, y, mu)
        weight = 1.0 / phi

        resid_covar = self.std_err(self.family, X, y, eta, mu, weight, alpha)
        beta_se = jnp.sqrt(jnp.diag(resid_covar))

        df = X.shape[0] - X.shape[1]
        beta = beta.squeeze()  # (p,)
        stat = beta / beta_se

        pval_wald = t_cdf(-abs(stat), df) * 2

        return GLMState(
            beta,
            beta_se,
            stat,
            pval_wald,
            eta,
            mu,
            weight,
            n_iter,
            converged,
            resid_covar,
            resid,
            alpha,
        )


class GLM(eqx.Module):
    """ """

    family: ExponentialFamily = Gaussian()
    solver: LinearSolve = CholeskySolve()
    std_err: ErrVarEstimation = FisherInfoError()

    max_iter: int = 1000
    tol: float = 1e-3
    step_size: float = 1.0

    _init: _AbstractInit = eqx.field(init=False)

    def __post_init__(self):
        if isinstance(self.family, NegativeBinomial):
            # this is kind of architectually annoying. NB family shouldn't know about the GLM
            # and we need to init NB using Poisson GLM, so this is our hack/workaround for now.
            self._init = _NBInit(self.family, self.solver, self.std_err)
        else:
            self._init = _SimpleInit(self.family, self.solver, self.std_err)

        return

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike = 0.0,
        max_iter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 1.0,
    ) -> GLMState:
        """Fit GLM

        :param X: covariate data matrix (nxp)
        :param y: outcome vector (nx1)
        :param offset: offset (nx1)
        :param init: initial value for betas
        :param alpha_init: initial value for alpha in NB model, default to 0s
        :param se_estimator: estimator for standard error, default to fisher information
        :return: GLMState that contains model fitting result
        """

        # initialize eta and alpha
        init, alpha_init = self._init(X, y, offset, max_iter, tol, step_size)
        beta, n_iter, converged, alpha = irls(
            X, y, offset, init, self.family, self.solver, self.max_iter, self.tol, self.step_size, alpha_init
        )
        eta = X @ beta + offset
        mu = self.family.glink.inverse(eta)
        resid = (y - mu) * self.family.glink.deriv(mu)  # note: this is the working resid
        _, _, weight = self.family.calc_weight(X, y, eta, alpha)
        resid_covar = self.std_err(self.family, X, y, eta, mu, weight, alpha)
        beta_se = jnp.sqrt(jnp.diag(resid_covar))
        beta = beta
        stat = beta / beta_se

        pval_wald = 2 * norm.sf(jnp.abs(stat))

        return GLMState(
            beta,
            beta_se,
            stat,
            pval_wald,
            eta,
            mu,
            weight,
            n_iter,
            converged,
            resid_covar,
            resid,
            alpha,
        )
