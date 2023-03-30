import numpy.testing as nptest
from statsmodels.genmod.generalized_linear_model import GLMResults

from jaxqtl.infer.glm import GLMState


def assert_betas_eq(state: GLMState, sm_state: GLMResults, rtol=1e-5):
    nptest.assert_allclose(state.beta, sm_state.params, rtol=rtol)


def assert_beta_array_eq(state: GLMState, sm_state: GLMState, rtol=1e-5):
    nptest.assert_allclose(state.beta, sm_state.beta, rtol=rtol)
