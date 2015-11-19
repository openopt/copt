import numpy as np
from scipy import linalg, optimize
from sklearn.linear_model import logistic
from gdprox import fmin_cgprox

n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():

    def logloss(x):
        return logistic._logistic_loss(x, X, y, 1.)

    def fprime_logloss(x):
        return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]

    def g_prox(x, alpha):
        return x
    opt = fmin_cgprox(
        logloss, fprime_logloss, g_prox, np.zeros(n_features),
        rtol=1e-12)
    out = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features), fprime=fprime_logloss)
    assert linalg.norm(out[0] - opt.x) < 1e-3


def test_sklearn():
    for alpha in np.logspace(-3, 3):

        def logloss(x):
            return logistic._logistic_loss(x, X, y, 0.)

        def fprime_logloss(x):
            return logistic._logistic_loss_and_grad(x, X, y, 0.)[1]

        def g_prox(x, step_size):
            """
            L1 regularization
            """
            return np.fmax(x - step_size * alpha, 0) - \
                np.fmax(- x - step_size * alpha, 0)

        clf = logistic.LogisticRegression(
            penalty='l1', fit_intercept=False, C=1 / alpha)
        clf.fit(X, y)
        opt = fmin_cgprox(
            logloss, fprime_logloss, g_prox, np.zeros(n_features),
            rtol=1e-12)
        assert linalg.norm(opt.x - clf.coef_) < 1e-3

