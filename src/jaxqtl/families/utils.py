import jax
import jax.numpy as jnp
from jax.scipy.special import expit


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
