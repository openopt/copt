import numpy as np
from scipy import optimize
from sklearn.linear_model import logistic
from copt.stochastic import saga

np.random.seed(0)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():

    alpha = 0.
    def logloss(x):
        return logistic._logistic_loss(x, X, y, alpha)

    def fprime_logloss(x):
        return logistic._logistic_loss_and_grad(x, X, y, alpha)[1]

    opt = saga(X, y, np.zeros(n_features), 'log', 1e-3)
    assert opt.success
    sol_scipy = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features), fprime=fprime_logloss)[0]
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)

    def squaredloss(x):
        return 0.5 * (y - np.dot(X, x)) ** 2 + 0.5 * alpha * x.dot(x)

    def fprime_squaredloss(x):
        return - X.T.dot(y - np.dot(X, x)) + alpha * x

    opt = saga(X, y, np.zeros(n_features), 'squared', 1e-3,
               max_iter=10000)
    assert opt.success
    sol_scipy = optimize.fmin_l_bfgs_b(
        squaredloss, np.zeros(n_features), fprime=fprime_squaredloss)[0]
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-6)
