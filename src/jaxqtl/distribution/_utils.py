import jax.numpy as jnp

from jax import lax
from jax.scipy.special import betainc, gammaincc
from jaxtyping import Array, ArrayLike


def t_cdf(value: ArrayLike, df: ArrayLike, loc: ArrayLike = 0.0, scale: ArrayLike = 1.0) -> Array:
    r"""Evaluate the cumulative distribution function of Student's t distribution.

    The arguments follow JAX broadcasting rules. This implementation is written in
    terms of `jax.scipy.special.betainc` so it can be transformed by JAX.

    **Arguments:**

    - `value`: Points at which to evaluate the CDF.
    - `df`: Positive degrees of freedom.
    - `loc`: Distribution location. Defaults to 0.
    - `scale`: Positive distribution scale. Defaults to 1.

    **Returns:**

    CDF values with the broadcasted shape of `value`, `df`, `loc`, and `scale`.
    """
    # Ref: https://en.wikipedia.org/wiki/Student's_t-distribution#Related_distributions
    # X^2 ~ F(1, df) -> df / (df + X^2) ~ Beta(df/2, 0.5)
    value = jnp.asarray(value)
    df = jnp.asarray(df)
    loc = jnp.asarray(loc)
    scale = jnp.asarray(scale)
    scaled = (value - loc) / scale
    scaled_squared = scaled * scaled
    beta_value = df / (df + scaled_squared)

    # when scaled < 0, returns 0.5 * Beta(df/2, 0.5).cdf(beta_value)
    # when scaled > 0, returns 1 - 0.5 * Beta(df/2, 0.5).cdf(beta_value)
    return 0.5 * (1 + jnp.sign(scaled) - jnp.sign(scaled) * betainc(0.5 * df, 0.5, beta_value))


def ncx2_sf(
    x: ArrayLike,
    df: float,
    nc: float,
    terms: int = 100,
):
    r"""Approximate the survival function of a noncentral chi-squared distribution.

    The approximation truncates the Poisson-mixture representation after `terms`
    components. Larger noncentrality parameters may require more terms to capture
    the upper tail accurately.

    **Arguments:**

    - `x`: Points at which to evaluate the survival function.
    - `df`: Positive degrees of freedom.
    - `nc`: Non-negative noncentrality parameter.
    - `terms`: Number of Poisson-mixture components. Defaults to 100.

    **Returns:**

    Approximate survival probabilities with the same shape as `x`.
    """
    x = jnp.asarray(x)
    half_x = x / 2.0
    half_df = df / 2.0
    half_nc = nc / 2.0
    log_half_nc = jnp.log(half_nc)

    # Initial state: (running_sum, log_weight, j_index)
    init = (
        jnp.zeros_like(x),  # sum over j
        -half_nc,  # log w_j at j=0
        0,
    )  # j

    def body(state, _):
        sum_so_far, log_w, j = state
        # central survival term Q(df/2 + j, x/2)
        surv = gammaincc(half_df + j, half_x)
        sum_so_far = sum_so_far + jnp.exp(log_w) * surv
        # update log_w -> log_w + log(half_nc) - log(j+1)
        log_w = log_w + log_half_nc - jnp.log(j + 1)
        return (sum_so_far, log_w, j + 1), None

    (final_sum, _, _), _ = lax.scan(body, init, None, length=terms)
    return final_sum
