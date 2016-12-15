import numpy as np
from scipy import optimize
from sklearn.linear_model import logistic
from copt.stochastic import saga

np.random.seed(0)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():

    L = np.max((X * X).sum(1))
    step_size = 1.0 / L

    alpha = 0.
    def logloss(x):
        return logistic._logistic_loss(x, X, y, alpha)

    def fprime_logloss(x):
        return logistic._logistic_loss_and_grad(x, X, y, alpha)[1]

    opt = saga('log', None, X, y, np.zeros(n_features))
    assert opt.success
    sol_scipy = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features), fprime=fprime_logloss)[0]
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)

    def squaredloss(w):
        return 0.5 * ((y - np.dot(X, w)) ** 2).sum() + 0.5 * alpha * w.dot(w)

    def fprime_squaredloss(w):
        return - X.T.dot(y - np.dot(X, w)) + alpha * w

    opt = saga('squared', None, X, y, np.zeros(n_features))
    assert opt.success
    print(fprime_squaredloss(opt.x))
    sol_scipy = optimize.fmin_l_bfgs_b(
        squaredloss, np.zeros(n_features), fprime=fprime_squaredloss)[0]
    print(fprime_squaredloss(sol_scipy))
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)
