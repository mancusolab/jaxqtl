import pytest

import jax
import jax.numpy as jnp
import jax.random as rdm

from jax import config
from jax.scipy.special import gammaln, xlogy

from jaxqtl.distribution._expfam import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson
from jaxqtl.distribution._links import IdentityLink, InverseLink, LogitLink, LogLink, NBLink, PowerLink


config.update("jax_enable_x64", True)


def test_poisson_negloglikelihood_accepts_fractional_response():
    y = jnp.asarray([0.0, 1.5, 3.25])
    mu = jnp.asarray([1.0, 2.0, 4.0])

    actual = Poisson().negloglikelihood(jnp.empty((y.size, 0)), y, jnp.log(mu), 1.0)
    expected = jnp.sum(mu - xlogy(y, mu) + gammaln(y + 1.0))

    assert jnp.isfinite(actual)
    assert float(actual) == pytest.approx(float(expected))


@pytest.mark.parametrize(
    ("link", "mu"),
    [
        (IdentityLink(), jnp.linspace(-2.0, 2.0, 101)),
        (LogLink(), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (InverseLink(), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (LogitLink(), jnp.linspace(0.05, 0.95, 101)),
        (PowerLink(0.5), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (PowerLink(2.0), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (NBLink(0.3), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
    ],
)
def test_link_roundtrip_inverse(link, mu):
    eta = link(mu)
    mu_back = link.inverse(eta)
    assert jnp.all(jnp.isfinite(eta))
    assert jnp.all(jnp.isfinite(mu_back))
    assert jnp.allclose(mu_back, mu, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("link", "mu"),
    [
        (IdentityLink(), jnp.linspace(-2.0, 2.0, 101)),
        (LogLink(), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (InverseLink(), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (LogitLink(), jnp.linspace(0.05, 0.95, 101)),
        (PowerLink(1.5), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
        (NBLink(0.3), jnp.exp(jnp.linspace(-2.0, 2.0, 101))),
    ],
)
def test_link_derivative_matches_autodiff(link, mu):
    mu = jnp.asarray(mu)

    def _forward(x):
        return link(x)

    deriv_ad = jax.vmap(jax.grad(lambda x: _forward(x).astype(float)))(mu.astype(float))
    deriv_impl = link.deriv(mu)
    assert jnp.all(jnp.isfinite(deriv_ad))
    assert jnp.all(jnp.isfinite(deriv_impl))
    assert jnp.allclose(deriv_impl, deriv_ad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("family", "disp"),
    [
        (Gaussian(), 1.0),
        (Poisson(), 1.0),
        (Binomial(), 1.0),
        (NegativeBinomial(), 0.2),
        (Gamma(), 0.7),
    ],
)
def test_family_calc_weight_shapes_and_finiteness(family, disp):
    eta = jnp.linspace(-1.0, 1.0, 128)
    mu, link_prime, weight = family.calc_weight(eta, disp)
    assert mu.shape == eta.shape
    assert link_prime.shape == eta.shape
    assert jnp.shape(weight) in [(), eta.shape]
    assert jnp.all(jnp.isfinite(mu))
    assert jnp.all(jnp.isfinite(link_prime))
    assert jnp.all(jnp.isfinite(jnp.atleast_1d(weight)))


@pytest.mark.parametrize(
    ("family", "disp"),
    [
        (Gaussian(), 1.3),
        (Poisson(), 1.0),
        (Binomial(), 1.0),
        (NegativeBinomial(), 0.4),
        (Gamma(), 0.6),
    ],
)
def test_family_sample_shapes_and_mean_approx(family, disp):
    # Monte Carlo check that E[y] ≈ mu using a fixed PRNG stream.
    if isinstance(family, Gaussian):
        mu = jnp.linspace(-1.0, 1.0, 64)
    elif isinstance(family, Binomial):
        mu = jnp.linspace(0.1, 0.9, 64)
    else:
        mu = jnp.exp(jnp.linspace(-1.0, 1.0, 64))

    eta = family.glink(mu)

    base_key = rdm.key(0)
    keys = rdm.split(base_key, 1000)
    ys = jax.vmap(lambda k: family.sample(k, eta, disp), in_axes=0)(keys)

    assert ys.shape == (keys.shape[0],) + eta.shape
    assert jnp.all(jnp.isfinite(ys))

    mean_hat = jnp.mean(ys, axis=0)

    # Deterministic, but allow generous tolerance for discrete outcomes.
    assert jnp.allclose(mean_hat, mu, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    ("family_ctor", "bad_link"),
    [
        (Gaussian, LogitLink()),
        (Poisson, LogitLink()),
        (Binomial, InverseLink()),
        (NegativeBinomial, InverseLink()),
        (Gamma, LogitLink()),
    ],
)
def test_invalid_links_raise_value_error(family_ctor, bad_link):
    with pytest.raises(ValueError):
        family_ctor(glink=bad_link)
