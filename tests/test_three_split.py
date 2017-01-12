import numpy as np
from copt import fmin_DavisYin, fmin_ProxGrad
from copt import prox

from sklearn.linear_model import logistic

np.random.seed(0)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))

# helper functions
def logloss(x):
    return logistic._logistic_loss(x, X, y, 1.)

def fprime_logloss(x):
    return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]

def fused_lasso(x):
    return np.abs(np.diff(x)).sum()


def test_optimize():

    opt = fmin_DavisYin(
        logloss, fprime_logloss, None, None,
        np.zeros(n_features), tol=1e-12)

    np.testing.assert_almost_equal(
        fprime_logloss(opt.x), np.zeros(n_features))


def test_L1():
    """
    Test that it coincides with solution provided by proximal
    gradient descent for L1-penalized logistic regression.
    """

    for alpha in np.logspace(-3, 3, 5):
        x0 = np.zeros(n_features)

        opt = fmin_DavisYin(
            logloss, fprime_logloss, prox.prox_L1, None,
            x0.copy(), alpha=alpha, beta=alpha)

        opt2 = fmin_DavisYin(
            logloss, fprime_logloss, None, prox.prox_L1,
            x0.copy(), alpha=alpha, beta=alpha)

        opt3 = fmin_ProxGrad(
            logloss, fprime_logloss, prox.prox_L1, x0.copy(), alpha=alpha)

        np.testing.assert_almost_equal(opt.x, opt2.x, decimal=2)
        np.testing.assert_almost_equal(opt2.x, opt3.x, decimal=2)


def test_fused():
    """Solve the a problem with a fused lasso penalty in two
    different, but equivalent ways:
    1. Using a proximal gradient descent penalty and
    the proximal operator of the fused lasso penalty.
    2. Decomposing the fused lasso penalty in a sum of two
    proximable penalties and using the three operator splitting.
    """
    def logloss(x):
        return logistic._logistic_loss(x, X, y, 1.)

    def fprime_logloss(x):
        return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]

    def g_prox(x, step_size):
        n_rows = x.size // 2
        Lx = np.empty(n_rows)
        for i in range(n_rows):
            Lx[i] = x[2 * i] - x[2 * i + 1]
        z = prox.prox_L1(Lx, 2 * step_size) - Lx
        tmp = np.zeros(x.size)
        for i in range(n_rows):
            tmp[2 * i] = z[i]
            tmp[2 * i + 1] = - z[i]
        return x + tmp / 2

    def h_prox(x, step_size):
        tmp = x.copy()
        tmp[1:] = g_prox(x[1:], step_size)
        return tmp

    for alpha in np.logspace(-3, 3, 5):
        x0 = np.zeros(n_features)
        opt1 = fmin_DavisYin(
            logloss, fprime_logloss, h_prox, g_prox, x0.copy(),
            alpha=alpha, beta=alpha)
        opt2 = fmin_ProxGrad(
            logloss, fprime_logloss, prox.prox_tv1d, x0.copy(),
            alpha=alpha)

        np.testing.assert_almost_equal(opt1.x, opt2.x)

