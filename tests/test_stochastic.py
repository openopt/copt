import numpy as np
from scipy import optimize, sparse
from sklearn.linear_model import logistic
from copt.stochastic import fmin_SAGA

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

    opt = fmin_SAGA('log', None, X, y, np.zeros(n_features))
    assert opt.success
    sol_scipy = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features), fprime=fprime_logloss)[0]
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)

    def squaredloss(w):
        return 0.5 * ((y - np.dot(X, w)) ** 2).sum() + 0.5 * alpha * w.dot(w)

    def fprime_squaredloss(w):
        return - X.T.dot(y - np.dot(X, w)) + alpha * w

    opt = fmin_SAGA('squared', None, X, y, np.zeros(n_features))
    assert opt.success
    print(fprime_squaredloss(opt.x))
    sol_scipy = optimize.fmin_l_bfgs_b(
        squaredloss, np.zeros(n_features), fprime=fprime_squaredloss)[0]
    print(fprime_squaredloss(sol_scipy))
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)


def test_sparse():
    # test with a sparse matrix
    Xs = sparse.random(n_samples, n_features, density=0.2, format='csr')
    opt = fmin_SAGA('log', None, Xs, y, np.zeros(n_features))
    opt2 = fmin_SAGA('log', None, Xs.toarray(), y, np.zeros(n_features))
    np.testing.assert_allclose(opt.x, opt2.x, rtol=1e-2)
