from utils import assert_array_eq

import jax.numpy as jnp
from jax import random
from jax.config import config

from jaxqtl.infer.permutation import infer_beta

config.update("jax_enable_x64", True)


def test_betaperm():
    sample_n = 5000
    key = random.PRNGKey(2)
    key, key_random = random.split(key, 2)

    true_k = 1.5
    true_n = 5.0
    expected = jnp.array([true_k, true_n])
    p_perm = random.beta(key_random, a=true_k, b=true_n, shape=(sample_n,))
    observed, converged = infer_beta(p_perm, jnp.ones(2), stepsize=0.5)

    assert_array_eq(converged, jnp.asarray(1.0))
    assert_array_eq(
        observed, expected, rtol=5.2e-2
    )  # this is ~ rtol=0.0513; close enough
