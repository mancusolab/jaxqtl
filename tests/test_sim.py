import pytest

import jax.numpy as jnp
import jax.random as rdm

from jaxqtl.families.distribution import Gaussian, Poisson
from jaxqtl.sim import simulate_pheno


@pytest.mark.parametrize(
    "family,gamma,dispersion",
    [
        (Gaussian(), 1.1, 2.0),
        (Poisson(), 0.0, None),
    ],
)
def test_simulate_shapes_and_eta(family, gamma, dispersion):
    key = rdm.PRNGKey(0)
    n, p = 8, 3
    beta = jnp.array([0.2, -0.3, 0.5])

    result = simulate_pheno(
        key,
        n=n,
        p=p,
        family=family,
        beta=beta,
        gamma=gamma,
        maf=0.2,
        dispersion=dispersion,
    )

    assert result.X.shape == (n, p)
    assert result.g.shape == (n,)
    assert result.offset.shape == (n,)
    assert result.y.shape == (n,)

    expected_eta = result.offset + result.X @ result.beta + result.gamma * result.g
    expected_mu = family.glink.inverse(expected_eta)
    assert jnp.allclose(result.eta, expected_eta)
    assert jnp.allclose(result.mu, expected_mu)


def test_simulate_poisson_mean_matches_mu():
    key = rdm.PRNGKey(1)
    beta = jnp.array([0.1, 0.2])
    data = simulate_pheno(
        key,
        n=2000,
        p=2,
        family=Poisson(),
        beta=beta,
        gamma=0.0,
        maf=0.3,
    )

    mu_mean = float(jnp.mean(data.mu))
    y_mean = float(jnp.mean(data.y))

    # Poisson mean should concentrate around mu; allow 10% relative tolerance
    assert abs(y_mean - mu_mean) / max(mu_mean, 1e-8) < 0.1
