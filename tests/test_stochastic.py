import numpy as np
from scipy import optimize, sparse
from sklearn.linear_model import logistic
from copt import fmin_SAGA, fmin_PSSAGA, fmin_PGD
from copt import prox, stochastic

np.random.seed(0)
n_samples, n_features = 100, 10
X_dense = np.random.randn(n_samples, n_features)
X_sparse = sparse.random(n_samples, n_features, density=0.2, format='csr')
y = np.sign(np.random.randn(n_samples))


def logloss(x):
    return logistic._logistic_loss(x, X_dense, y, 0.0) / n_samples


def fprime_logloss(x):
    return logistic._logistic_loss_and_grad(x, X_dense, y, 0.0)[1] / n_samples


def test_optimize():

    step_size = stochastic.compute_step_size('logistic', X_dense)
    opt = fmin_SAGA(
        stochastic.f_logistic, stochastic.deriv_logistic,
        X_dense, y, np.zeros(n_features), step_size=step_size,
        trace=True)
    assert opt.success
    sol_scipy = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features), fprime=fprime_logloss)[0]
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)
    np.testing.assert_allclose([logloss(opt.x)], [opt.trace_fun[-1]])

    alpha = 0
    def squaredloss(w):
        return 0.5 * ((y - np.dot(X_dense, w)) ** 2).sum() + 0.5 * alpha * w.dot(w)

    def fprime_squaredloss(w):
        return - X_dense.T.dot(y - np.dot(X_dense, w)) + alpha * w

    step_size = stochastic.compute_step_size('squared', X_dense)
    opt = fmin_SAGA(
        stochastic.f_squared, stochastic.deriv_squared,
        X_dense, y, np.zeros(n_features), trace=True, step_size=step_size)
    assert opt.success
    opt2 = fmin_PSSAGA(stochastic.f_squared, stochastic.deriv_squared,
                       X_dense, y, None, None, np.zeros(n_features),
                       step_size=step_size, trace=True)
    assert opt.success
    print(fprime_squaredloss(opt.x))
    sol_scipy = optimize.fmin_l_bfgs_b(
        squaredloss, np.zeros(n_features), fprime=fprime_squaredloss)[0]
    print(fprime_squaredloss(sol_scipy))
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)
    np.testing.assert_allclose(sol_scipy, opt2.x, rtol=1e-1)


def test_L1():
    for X in (X_dense, ):
        for beta in np.logspace(-3, 3, 5):
            step_size = stochastic.compute_step_size('logistic', X_dense)
            opt = fmin_SAGA(
                stochastic.f_logistic, stochastic.deriv_logistic,
                X, y, np.zeros(n_features), step_size=step_size,
                beta=beta, g_prox=prox.prox_L1, trace=True, verbose=True)

            def loss(x):
                return logistic._logistic_loss(x, X, y, 0.0) / n_samples

            def grad(x):
                return logistic._logistic_loss_and_grad(x, X, y, 0.0)[1] / n_samples

            opt2 = fmin_PGD(
                loss, grad, prox.prox_L1, np.zeros(n_features),
                alpha=beta)
            assert opt.success
            np.testing.assert_allclose(opt.x, opt2.x, rtol=1e-2)


def test_sparse():
    # test with a sparse matrix
    for g_prox in (None, prox.prox_L1):
        step_size = stochastic.compute_step_size('logistic', X_sparse)
        opt = fmin_SAGA(
            stochastic.f_logistic,
            stochastic.deriv_logistic, X_sparse, y, np.zeros(n_features),
            step_size=step_size, g_prox=g_prox)
        opt2 = fmin_SAGA(
            stochastic.f_logistic, stochastic.deriv_logistic,
            X_sparse.toarray(), y, np.zeros(n_features),
            step_size=step_size, g_prox=g_prox)
        np.testing.assert_allclose(opt.x, opt2.x, rtol=1e-2)


