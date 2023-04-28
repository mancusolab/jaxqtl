import numpy as np
import numpy.linalg as npla
import statsmodels.api as sm
from numpy import random

# simulate under logistic model
n = 100
random.seed(1)
X1 = random.normal(size=n).reshape((n, 1))  # covariate
g = random.normal(size=n).reshape((n, 1))  # variable of primary interest
intercept = 0.5
X = np.hstack([np.ones_like(X1), X1, g])
beta = np.array([intercept, 1.0, 2.0]).reshape([3, 1])
eta = X @ beta
prob = 1 / (1 + np.exp(-eta))
y = random.binomial(n=1, p=prob)

# fit full model y ~ X1 + X2
mod_full = sm.GLM(y, X, family=sm.families.Binomial()).fit()

# covariate only model: y ~ X1
covar = X[:, 0:2]
mod_covar = sm.GLM(y, covar, family=sm.families.Binomial()).fit()

mod_null_eta = mod_covar.get_prediction(covar, which="linear").predicted
mod_null_mu = mod_covar.get_prediction(covar, which="mean").predicted

glm_weight = mod_null_mu * (1 - mod_null_mu)
w_half_X = np.diag(np.sqrt(glm_weight)) @ covar
w_X = np.diag(glm_weight) @ covar

# regress out covar from X2
projection_covar = covar @ npla.inv(w_half_X.T @ w_half_X) @ w_X.T  # nxn
X2_resid = g - projection_covar @ g

# fit model y ~ X2_resid with offset
mod_X2 = sm.GLM(y, X2_resid, family=sm.families.Binomial(), offset=mod_null_eta).fit()

print(mod_full.summary())
print(mod_X2.summary())
