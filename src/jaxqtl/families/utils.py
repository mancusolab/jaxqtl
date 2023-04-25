from jaxtyping import ArrayLike

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, expit


def _clipped_expit(x):
    finfo = jnp.finfo(jnp.result_type(x))
    return jnp.clip(expit(x), a_min=finfo.tiny, a_max=1.0 - finfo.eps)


def _grad_per_sample(func, x):
    """Get gradient for each sample
    x.shape = (n,1), eg. x can be mu or eta
    need to convert x to (n,) in order to apply vmap and grad
    """
    n = x.shape[0]
    grad_fn = jax.vmap(jax.grad(func), 0)
    deriv_val = grad_fn(x.reshape((n,)))
    return deriv_val.reshape((n, 1))


def t_cdf(value: ArrayLike, df: float, loc: ArrayLike = 0.0, scale: ArrayLike = 1.0):
    # Ref: https://en.wikipedia.org/wiki/Student's_t-distribution#Related_distributions
    # X^2 ~ F(1, df) -> df / (df + X^2) ~ Beta(df/2, 0.5)
    scaled = (value - loc) / scale
    scaled_squared = scaled * scaled
    beta_value = df / (df + scaled_squared)

    # when scaled < 0, returns 0.5 * Beta(df/2, 0.5).cdf(beta_value)
    # when scaled > 0, returns 1 - 0.5 * Beta(df/2, 0.5).cdf(beta_value)
    return 0.5 * (
        1 + jnp.sign(scaled) - jnp.sign(scaled) * betainc(0.5 * df, 0.5, beta_value)
    )
