import numpy as np
from scipy import optimize, sparse
from sklearn.linear_model import logistic
from copt import fmin_SAGA, fmin_PSSAGA, fmin_PGD
from copt import prox

np.random.seed(0)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
Xs = sparse.random(n_samples, n_features, density=0.2, format='csr')
y = np.sign(np.random.randn(n_samples))


def logloss(x):
    return logistic._logistic_loss(x, X, y, 0.0) / n_samples


def fprime_logloss(x):
    return logistic._logistic_loss_and_grad(x, X, y, 0.0)[1] / n_samples


def test_optimize():

    opt = fmin_SAGA('logistic', None, X, y, np.zeros(n_features), trace=True)
    assert opt.success
    sol_scipy = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features), fprime=fprime_logloss)[0]
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)
    np.testing.assert_allclose([logloss(opt.x)], [opt.trace_fun[-1]])

    alpha = 0
    def squaredloss(w):
        return 0.5 * ((y - np.dot(X, w)) ** 2).sum() + 0.5 * alpha * w.dot(w)

    def fprime_squaredloss(w):
        return - X.T.dot(y - np.dot(X, w)) + alpha * w

    opt = fmin_SAGA('squared', None, X, y, np.zeros(n_features), trace=True)
    assert opt.success
    opt2 = fmin_PSSAGA('squared', None, X, y, None, None, np.zeros(n_features),
                       trace=True)
    assert opt.success
    print(fprime_squaredloss(opt.x))
    sol_scipy = optimize.fmin_l_bfgs_b(
        squaredloss, np.zeros(n_features), fprime=fprime_squaredloss)[0]
    print(fprime_squaredloss(sol_scipy))
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)
    np.testing.assert_allclose(sol_scipy, opt2.x, rtol=1e-1)


def test_L1():
    opt = fmin_SAGA('logistic', None, X, y, np.zeros(n_features),
                    g_prox=prox.prox_L1 , trace=True)

    beta = 0  # XXX FIXME
    opt2 = fmin_PGD(
        logloss, fprime_logloss, prox.prox_L1, np.zeros(n_features),
        alpha=beta, tol=1e-3)
    assert opt.success


def test_sparse():
    # test with a sparse matrix
    opt = fmin_SAGA('logistic', None, Xs, y, np.zeros(n_features))
    opt2 = fmin_SAGA('logistic', None, Xs.toarray(), y, np.zeros(n_features))
    np.testing.assert_allclose(opt.x, opt2.x, rtol=1e-2)

    # XXX test with L1


