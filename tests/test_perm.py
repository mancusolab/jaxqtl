from utils import assert_array_eq

import jax.numpy as jnp

from jax import random
from jax.config import config

from jaxqtl.infer.permutation import infer_beta


config.update("jax_enable_x64", True)


def test_betaperm():
    sample_n = 500
    key = random.PRNGKey(2)
    key, key_random = random.split(key, 2)

    true_k = 1.5
    true_n = 1000.0
    expected = jnp.array([true_k, true_n])
    p_perm = random.beta(key_random, a=true_k, b=true_n, shape=(sample_n,))

    # init = jnp.ones(2)
    p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
    k_init = p_mean * (p_mean * (1 - p_mean) / p_var - 1)
    n_init = k_init * (1 / p_mean - 1)
    init = jnp.array([k_init, n_init])

    res = infer_beta(p_perm, init, step_size=1.0)

    observed = res[0:2]
    converged = res[2]

    print(f"truth: {true_k}, {true_n}")
    print(f"observed: {observed}; converged: {converged}")

    assert_array_eq(converged, jnp.asarray(1.0))
    assert_array_eq(observed, expected, rtol=5.2e-2)  # this is ~ rtol=0.0513; close enough
