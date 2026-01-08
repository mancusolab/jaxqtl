from utils import assert_array_eq

import jax.numpy as jnp

from jax import config, random

from jaxqtl.infer.optimize import infer_beta_params


config.update("jax_enable_x64", True)


def test_betaperm():
    sample_n = 500
    key = random.PRNGKey(2)
    key, key_random = random.split(key, 2)

    true_k = 1.5
    true_n = 1000.0
    p_perm = random.beta(key_random, a=true_k, b=true_n, shape=(sample_n,))

    # init = jnp.ones(2)
    p_mean, p_var = jnp.mean(p_perm), jnp.var(p_perm)
    k_init = p_mean * (p_mean * (1 - p_mean) / p_var - 1)
    n_init = k_init * (1 / p_mean - 1)
    init = jnp.array([k_init, n_init])

    res = infer_beta_params(p_perm, init, step_size=1.0)

    k_hat, n_hat, converged = res

    print(f"truth: {true_k}, {true_n}")
    print(f"observed: {(k_hat, n_hat)}; converged: {converged}")

    assert_array_eq(converged, jnp.asarray(1.0))

    assert k_hat > 0
    assert n_hat > 0

    # Finite-sample MLE will not recover exact latent parameters; instead verify fit improves and moments match.
    def _loglik(k, n, p):
        from jax.scipy import stats as jaxstats

        return jnp.sum(jaxstats.beta.logpdf(p, k, n))

    init_lik = _loglik(init[0], init[1], p_perm)
    final_lik = _loglik(k_hat, n_hat, p_perm)
    assert final_lik > init_lik

    mean_emp = jnp.mean(p_perm)
    var_emp = jnp.var(p_perm)

    mean_hat = k_hat / (k_hat + n_hat)
    var_hat = (k_hat * n_hat) / ((k_hat + n_hat) ** 2 * (k_hat + n_hat + 1))

    assert_array_eq(mean_hat, mean_emp, rtol=5e-2)
    assert_array_eq(var_hat, var_emp, rtol=2e-1)
