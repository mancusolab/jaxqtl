# Variant-level tests

`jaxqtl` provides variant-level association tests for scanning genotypes against molecular phenotypes.

## Initialize an outcome and test variants

All concrete tests implement `AbstractHypothesisTest[StateT]`. `init(X, y, offset)` prepares an outcome, and
`test(X, G, state)` tests variants against that state. The rows of `X`, `G`, `y`, and a vector offset must describe
the same samples in the same order. Genotypes occupy the columns of `G`.

This example fits a Poisson null model once and tests two genotype blocks:

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from jaxqtl.distribution import Poisson
from jaxqtl.hypothesis import ScoreTest
from jaxqtl.infer import GeneralizedLinearModel

jax.config.update("jax_enable_x64", True)
X = jnp.ones((8, 1))  # Intercept-only null model.
y = jnp.array([2., 1., 4., 3., 5., 6., 7., 2.])
G = jnp.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.],
               [1., 2.], [2., 1.], [2., 2.], [0., 2.]])

test = ScoreTest(model=GeneralizedLinearModel(family=Poisson()))
initialize = eqx.filter_jit(test.init)
test_block = eqx.filter_jit(test.test)
state = initialize(X, y, 0.0)
first = test_block(X, G[:, :1], state)
second = test_block(X, G[:, 1:], state)
pvalues = jnp.concatenate((first.p, second.p))
```

For a single array, `test(X, G, y, offset)` combines initialization and testing and returns a `TestResult`.
Pass the same covariate matrix to initialization and testing. Reinitialize when the phenotype, offset, covariates,
or test configuration changes.

| Test | Initialization state | Work performed for each genotype block |
| --- | --- | --- |
| `ScoreTest` | `ScoreState`: null-model residuals, weights, and fit diagnostics | Residualize genotypes and calculate score statistics |
| `SpaTest` | `SpaState`: score state, link derivatives, and CGF state | Calculate scores and evaluate saddlepoint tails |
| `WaldTest` with `LinearModel` | `GaussianWaldState`: null residuals and weights | Fit variant effects using the residualized linear model |
| `WaldTest` with a GLM | `GlmWaldState`: response and offset | Fit a full model containing each tested variant |

States are fixed-structure JAX PyTrees and exclude the shared covariate matrix and genotypes. `WaldState` is the
union of the two Wald state types. GLM Wald reuses the compiled fitting function; it still fits a separate model
for each variant.

The CLI's mapping executor compiles initialization separately from fixed-size test blocks and handles padding
internally. Direct array callers control their own JIT boundaries and block shapes. Different array shapes can
require separate compilations.

## Implement a hypothesis test

Subclass `AbstractHypothesisTest[StateT]` with the concrete state type. Implement `init`, `test`, and the `name`
property, and provide `model` and `std_err`. Initialization must be independent of genotype-window width. Both
methods must support JAX transformations; file loading, block packing, and output writing belong to mapping.

??? abstract "`jaxqtl.hypothesis.AbstractHypothesisTest`"

    ::: jaxqtl.hypothesis.AbstractHypothesisTest
        options:
            members:
                - __init__
                - __call__
                - init
                - test
---

## Score and Wald tests

::: jaxqtl.hypothesis.ScoreTest
    options:
        members:
            - __init__
            - init
            - test

---

::: jaxqtl.hypothesis.WaldTest
    options:
        members:
            - __init__
            - init
            - test

---

## Saddlepoint approximation

SPA starts from the score test's null fit. Choose a CGF matching the model family. It uses bisection with
finite, sign-changing brackets constructed inside the CGF domain. The normal approximation is used when SPA
is not attempted under the score cutoff and support checks. An attempted SPA calculation that does not
converge or yields an invalid correction returns NaN; ACAT propagates NaN inputs. The returned `converged`
field describes model fitting, not whether SPA was applied successfully.

::: jaxqtl.hypothesis.SpaTest
    options:
        members:
            - __init__
            - init
            - test

---


??? abstract "`jaxqtl.hypothesis.CumulantGeneratingFunction`"

    ::: jaxqtl.hypothesis.CumulantGeneratingFunction
        options:
            members:
                - init
                - get_score_bounds
                - get_t_bounds
                - cgf

---

::: jaxqtl.hypothesis.GaussianCGF
    options:
        members: false

---

::: jaxqtl.hypothesis.NegativeBinomialCGF
    options:
        members: false

---

::: jaxqtl.hypothesis.PoissonCGF
    options:
        members: false

---

`saddlepoint_pvalue` accepts `ScalarLike` values for `score` and `scale`, and an `ArrayLike` vector of
residualized genotypes with shape `(n,)` for `g_resid`. Scalar inputs may be Python scalars or scalar arrays.
`CGFStateT` denotes the state type returned by the selected `CumulantGeneratingFunction.init`; pass that state
as `state` when evaluating the tail probability.

::: jaxqtl.hypothesis.saddlepoint_pvalue

---

## Result type

::: jaxqtl.hypothesis.TestResult
    options:
        members: false

## Initialization states

::: jaxqtl.hypothesis.ScoreState
    options:
        members: false

::: jaxqtl.hypothesis.SpaState
    options:
        members: false

::: jaxqtl.hypothesis.GaussianWaldState
    options:
        members: false

::: jaxqtl.hypothesis.GlmWaldState
    options:
        members: false
