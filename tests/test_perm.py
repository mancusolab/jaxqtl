from utils import assert_array_eq

import jax.numpy as jnp
from jax import random
from jax.config import config

from jaxqtl.infer.permutation import BetaPerm

config.update("jax_enable_x64", True)


def test_betaperm():

    sample_n = 50000

    key = random.PRNGKey(2)
    key, key_random = random.split(key, 2)

    true_k = 1.5
    true_n = 5

    p_perm = random.beta(key_random, a=true_k, b=true_n, shape=(sample_n,))
    k, n, converged = BetaPerm._infer_beta(p_perm, jnp.array([1.0, 1.0]))

    assert_array_eq(jnp.array([k, n]), jnp.array([true_k, true_n]))
