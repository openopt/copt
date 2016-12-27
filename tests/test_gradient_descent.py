import numpy as np
from scipy import optimize
from sklearn.linear_model import logistic
from copt import two_prox_grad
from nose import tools

np.random.seed(0)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():

    def logloss(x):
        return logistic._logistic_loss(x, X, y, 1.)

    def fprime_logloss(x):
        return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]


    # check that it rases exception when max_iter_backtracking
    # is negative
    tools.assert_raises(ValueError,
                        two_prox_grad, logloss, fprime_logloss, None,
                        np.zeros(n_features), max_iter_backtracking=-1)

    opt = two_prox_grad(
        logloss, fprime_logloss, None, np.zeros(n_features),
        tol=1e-3)
    assert opt.success
    sol_scipy = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features), fprime=fprime_logloss)[0]
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)


def test_sklearn():
    for alpha in np.logspace(-3, 3, 3):

        def logloss(x):
            return logistic._logistic_loss(x, X, y, 0.)

        def fprime_logloss(x):
            return logistic._logistic_loss_and_grad(x, X, y, 0.)[1]

        def g_prox(x, step_size):
            """
            L1 proximal operator
            """
            return np.fmax(x - step_size * alpha, 0) - \
                np.fmax(- x - step_size * alpha, 0)

        clf = logistic.LogisticRegression(
            penalty='l1', fit_intercept=False, C=1 / alpha)
        clf.fit(X, y)
        opt = two_prox_grad(
            logloss, fprime_logloss, g_prox, np.zeros(n_features),
            tol=1e-3)
        assert opt.success
        np.testing.assert_allclose(clf.coef_.ravel(), opt.x, rtol=1e-1)


