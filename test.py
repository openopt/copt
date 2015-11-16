import numpy as np
from scipy import linalg, optimize
from sklearn.linear_model import logistic
from gdprox import fmin_cgprox

n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))
alpha = 0.


def test_optimize():

    def logloss(x):
        return logistic._logistic_loss(x, X, y, 1.)

    def fprime_logloss(x):
        return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]

    def g_prox(x, alpha):
        return x
    w_ = fmin_cgprox(
        logloss, fprime_logloss, g_prox, np.zeros(n_features), verbose=True,
        rtol=1e-12)
    out = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features), fprime=fprime_logloss)
    assert linalg.norm(out[0] - w_) < 1e-3


# def test_sklearn():
#     def g_prox(x, step_size):
#         """
#         prox of el1
#         """
#         return np.fmax(x - step_size * alpha, 0) - \
#             np.fmax(- x - step_size * alpha, 0)

#     clf = logistic.LogisticRegression(
#         penalty='l2', fit_intercept=False, C=1e100, solver='lbfgs')
#     clf.fit(X, y)
#     w_ = fmin_cgprox(
#         logloss, fprime_logloss, g_prox, np.zeros(n_features), verbose=True)
#     assert linalg.norm(w_ - clf.coef_) < 1e-3

