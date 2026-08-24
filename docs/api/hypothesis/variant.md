# Variant-level tests

`jaxqtl` provides variant-level association tests for scanning genotypes against molecular phenotypes.

??? abstract "`jaxqtl.hypothesis.AbstractHypothesisTest`"

    ::: jaxqtl.hypothesis.AbstractHypothesisTest
        options:
            members:
                - __init__
                - __call__
                - test
---

## Score and Wald tests

::: jaxqtl.hypothesis.ScoreTest
    options:
        members: false

---

::: jaxqtl.hypothesis.WaldTest
    options:
        members: false

---

## Saddlepoint approximation

::: jaxqtl.hypothesis.SpaTest
    options:
        members:
            - __init__

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

::: jaxqtl.hypothesis.saddlepoint_pvalue

---

## Result type

::: jaxqtl.hypothesis.TestResult
    options:
        members: false
