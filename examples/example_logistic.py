"""
L1-regularized logistic regression
==================================

Implementation of L1-regularized logistic regression
using copt.
"""
import numpy as np
from sklearn.linear_model import logistic
from copt import fmin_PGD, prox

n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)
alpha = 1.


def logloss(x):
    return logistic._logistic_loss(x, X, y, 1.)


def fprime_logloss(x):
    return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]

out = fmin_PGD(logloss, fprime_logloss, prox.prox_L1, np.zeros(n_features))
print('Solution', out)
