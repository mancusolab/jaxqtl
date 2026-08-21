# pattern: Imperative Shell

import os
import subprocess
import sys

from pathlib import Path


_X32_WORKER_FLAG = "--_run-nb-x32"


def _run_negative_binomial_x32_case() -> None:
    # These imports must stay inside the child worker. Importing JAX while pytest
    # collects this module would make the process-global precision setting impure.
    import numpy as np

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from jaxqtl.distribution._expfam import NegativeBinomial
    from jaxqtl.infer._glm import GeneralizedLinearModel

    assert not jax.config.read("jax_enable_x64")

    rng = np.random.default_rng(18)
    n = 160
    X = np.column_stack((np.ones(n), rng.normal(size=(n, 2))))
    beta = np.array([0.35, 0.2, -0.15])
    offset = np.linspace(-0.35, 0.3, n) + 0.1 * np.sin(np.linspace(0.0, 3.0 * np.pi, n))
    eta = X @ beta + offset
    mu = np.exp(eta)
    alpha = 0.35
    size = 1.0 / alpha
    y = rng.negative_binomial(size, size / (size + mu))
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)

    max_iter = 200
    model = GeneralizedLinearModel(family=NegativeBinomial(), max_iter=max_iter, tol=1e-4)
    eager = model.fit(X, y, offset)
    jitted = eqx.filter_jit(model.fit)(X, y, offset)

    numerical_fields = (
        "beta",
        "se",
        "z",
        "p",
        "eta",
        "mu",
        "glm_wt",
        "link_prime",
        "resid_covar",
        "resid",
        "disp",
    )
    for state in (eager, jitted):
        assert bool(np.asarray(state.converged))
        assert np.isfinite(np.asarray(state.num_iters))
        assert 0 < int(np.asarray(state.num_iters)) <= max_iter
        assert np.asarray(state.disp) > 0.0
        for field in ("beta", "se", "z", "p"):
            assert getattr(state, field).shape == (X.shape[1],)
        for field in ("eta", "mu", "glm_wt", "link_prime", "resid"):
            assert getattr(state, field).shape == (n,)
        assert state.resid_covar.shape == (X.shape[1], X.shape[1])
        assert np.asarray(state.disp).shape == ()
        assert np.asarray(state.num_iters).shape == ()
        assert np.asarray(state.converged).shape == ()
        for field in numerical_fields:
            assert np.all(np.isfinite(np.asarray(getattr(state, field))))
        np.testing.assert_allclose(
            np.asarray(state.eta),
            np.asarray(X @ state.beta + offset),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(state.mu),
            np.exp(np.asarray(state.eta)),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(state.resid),
            (np.asarray(y) - np.asarray(state.mu)) / np.asarray(state.mu),
            rtol=1e-5,
            atol=1e-5,
        )

    # Outer JIT may change float32 reduction and fusion order across CPU backends.
    # This tolerance covers the cross-backend variation observed in the fixture.
    parity_rtol = 1e-3
    parity_atol = 1e-4
    for field in numerical_fields:
        np.testing.assert_allclose(
            np.asarray(getattr(jitted, field)),
            np.asarray(getattr(eager, field)),
            rtol=parity_rtol,
            atol=parity_atol,
            err_msg=f"eager/JIT mismatch for {field}",
        )
    # Float32 rounding can change the iteration where the stopping threshold is
    # crossed, so convergence is required above without equating iteration counts.


def test_negative_binomial_fit_with_offset_in_x32_process() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "0"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), _X32_WORKER_FLAG],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, (
        f"float32 negative-binomial worker failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


if __name__ == "__main__" and sys.argv[1:] == [_X32_WORKER_FLAG]:
    _run_negative_binomial_x32_case()
