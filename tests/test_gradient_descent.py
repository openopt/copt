import numpy as np
from scipy import linalg, optimize
from sklearn.linear_model import logistic
from copt import proximal_gradient

np.random.seed(0)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():

    def logloss(x):
        return logistic._logistic_loss(x, X, y, 1.)

    def fprime_logloss(x):
        return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]

    def g_prox(x, _):
        return x
    opt = proximal_gradient(
        logloss, fprime_logloss, g_prox, np.zeros(n_features),
        tol=1e-12)
    assert opt.success
    sol_scipy = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features), fprime=fprime_logloss)[0]
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)


def test_sklearn():
    for alpha in np.logspace(-3, 3, 5):

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
        opt = proximal_gradient(
            logloss, fprime_logloss, g_prox, np.zeros(n_features),
            tol=1e-6)
        assert opt.success
        assert linalg.norm(opt.x - clf.coef_) < 1e-3


