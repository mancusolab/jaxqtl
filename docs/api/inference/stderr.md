# Variance estimators

`jaxqtl` separates model fitting from the choice of coefficient covariance estimator. The available implementations
provide classical Fisher-information or Huber–White sandwich standard errors.

??? abstract "`jaxqtl.infer.AbstractVarianceEstimator`"

    ::: jaxqtl.infer.AbstractVarianceEstimator
        options:
            show_bases: true
            members:
                - __call__

## Covariance estimators

::: jaxqtl.infer.FisherInfoError
    options:
        show_bases: true
        members:
            - __call__

---

::: jaxqtl.infer.HuberError
    options:
        show_bases: true
        members:
            - __call__
