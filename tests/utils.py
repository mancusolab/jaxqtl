import numpy.testing as nptest

from jax.typing import ArrayLike


def assert_array_eq(estimate: ArrayLike, truth: ArrayLike, rtol=1e-5):
    nptest.assert_allclose(estimate, truth, rtol=rtol)
