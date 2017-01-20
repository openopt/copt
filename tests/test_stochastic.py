import numpy as np
from numba import njit
from scipy import optimize, sparse
from sklearn.linear_model import logistic
from copt import fmin_SAGA, fmin_PGD
from copt import stochastic
from copt import prox

np.random.seed(0)
n_samples, n_features = 100, 10
X_sparse = sparse.random(n_samples, n_features, density=0.1, format='csr')
#
# # remove samples that are zero
# # TODO: make it robust to this case
# idx = np.array(X_sparse.sum(1)).ravel() != 0
# X_sparse = X_sparse[idx]
# n_samples = X_sparse.shape[0]

X_dense = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))



def test_optimize():

    for alpha in np.logspace(-1, 3, 3):
        print(alpha)
        def logloss(x):
            return logistic._logistic_loss(x, X_dense, y, alpha * n_samples) / n_samples

        def fprime_logloss(x):
            return logistic._logistic_loss_and_grad(
                x, X_dense, y, alpha * n_samples)[1] / n_samples

        # now similar test byt with the squared loss instead
        def squaredloss(w):
            return 0.5 * ((y - np.dot(X_dense, w)) ** 2).sum() + 0.5 * alpha * n_samples * w.dot(w)

        def fprime_squaredloss(w):
            return - X_dense.T.dot(y - np.dot(X_dense, w)) + alpha * n_samples * w

        step_size = stochastic.compute_step_size('logistic', X_dense, alpha) / 4
        opt = stochastic.fmin_SAGA_fast(
            stochastic.f_logistic, stochastic.deriv_logistic,
            X_dense, y, np.zeros(n_features), step_size=step_size,
            alpha=alpha)
        # assert opt.success
        sol_scipy = optimize.fmin_l_bfgs_b(
            logloss, np.zeros(n_features), fprime=fprime_logloss)[0]

        np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)
        # np.testing.assert_allclose([logloss(opt.x)], [opt.trace_fun[-1]])

        step_size = stochastic.compute_step_size('squared', X_dense, alpha)
        opt = stochastic.fmin_SAGA_fast(
            stochastic.f_squared, stochastic.deriv_squared,
            X_dense, y, np.zeros(n_features), alpha=alpha, step_size=step_size)
        # assert opt.certificate < 1e-2
        opt2 = stochastic.fmin_PSSAGA_fast(
            stochastic.f_squared, stochastic.deriv_squared, X_dense, y,
            np.zeros(n_features), alpha=alpha, step_size=step_size / 2., tol=0)
        assert opt.certificate < 1e-2
        sol_scipy = optimize.fmin_l_bfgs_b(
            squaredloss, np.zeros(n_features), fprime=fprime_squaredloss)[0]
        # Compare to SciPy's LFBGS
        np.testing.assert_allclose(sol_scipy, opt.x, atol=1e-1)
        np.testing.assert_allclose(sol_scipy, opt2.x, atol=1e-1)


def test_prox_sparse():
    alpha = 1.0 / n_samples

    for X in (X_dense, X_sparse):
        def loss(x):
            return logistic._logistic_loss(x, X, y, 1.0) / n_samples

        def grad(x):
            return logistic._logistic_loss_and_grad(x, X, y, 1.0)[1] / n_samples

        step_size = stochastic.compute_step_size('logistic', X, alpha)
        for beta in np.logspace(-3, 3, 3):
            opt = stochastic.fmin_SAGA_fast(
                stochastic.f_logistic, stochastic.deriv_logistic,
                X, y, np.zeros(n_features), step_size=step_size,
                alpha=alpha, beta=beta, g_prox=stochastic.prox_L1)
            opt2 = fmin_PGD(
                loss, grad, prox.prox_L1, np.zeros(n_features),
                alpha=beta)
            # assert opt.success
            np.testing.assert_allclose(opt.x, opt2.x, atol=1e-1)

            opt3 = stochastic.fmin_PSSAGA_fast(
                stochastic.f_logistic, stochastic.deriv_logistic,
                X, y, np.zeros(n_features), step_size=step_size,
                alpha=alpha, gamma=beta, h_prox=stochastic.prox_L1, tol=0)
            opt4 = stochastic.fmin_PSSAGA_fast(
                stochastic.f_logistic, stochastic.deriv_logistic,
                X, y, np.zeros(n_features), step_size=step_size,
                alpha=alpha, beta=beta, g_prox=stochastic.prox_L1, tol=0)
            # np.testing.assert_allclose(opt.x, opt3.x, atol=1e-1)
            assert np.abs(loss(opt.x) - loss(opt3.x)) < 0.1
            assert np.abs(loss(opt.x) - loss(opt4.x)) < 0.1


def test_prox_groups():
    """Test sparse problems with group structure

    The nonsmooth term that we use is
        |x_1 - x_2| + |x_3 - x_4| + |x_5 - x_6|
    """

    alpha = 1.

    X_sparse = X_dense

    def loss(x):
        return logistic._logistic_loss(x, X_sparse, y, alpha * n_samples) / n_samples

    def grad(x):
        return logistic._logistic_loss_and_grad(x, X_sparse, y, alpha * n_samples)[1] / n_samples

    prox_L1 = njit(prox.prox_L1)

    def g_prox(step_size, y, low, high):
        x = y[low:high]
        n_rows = x.size // 2
        Lx = np.empty(n_rows)
        for i in range(n_rows):
            Lx[i] = x[2 * i] - x[2 * i + 1]
        z = prox_L1(2 * step_size, Lx) - Lx
        tmp = np.zeros(x.size)
        for i in range(n_rows):
            tmp[2 * i] = z[i]
            tmp[2 * i + 1] = - z[i]
        y[low:high] = x + tmp / 2
        return y.copy()

    def g_prox_stochastic(step_size, x, low, high):
        a = x[low] - x[low + 1]
        z = np.fmax(a - 2 * step_size, 0) - np.fmax(- a - 2 * step_size, 0) - a
        x[low] += z / 2.
        x[low + 1] -= z / 2.

    groups = np.arange(n_features) // 2
    step_size = stochastic.compute_step_size('logistic', X_sparse, alpha)
    for beta in np.logspace(-3, 3, 3):
        print(beta)

        opt = stochastic.fmin_SAGA_fast(
            stochastic.f_logistic, stochastic.deriv_logistic,
            X_sparse, y, np.zeros(n_features), step_size=step_size,
            beta=beta, g_prox=g_prox, g_blocks=groups, alpha=alpha, max_iter=10)
        opt2 = fmin_PGD(
            loss, grad, g_prox, np.zeros(n_features),
            alpha=beta, g_prox_args=(0, n_features))
        np.testing.assert_allclose(opt.x, opt2.x, atol=1e-1)

        # PSSAGA now!
        opt3 = stochastic.fmin_PSSAGA_fast(
            stochastic.f_logistic, stochastic.deriv_logistic,
            X_sparse, y, np.zeros(n_features), step_size=step_size,
            beta=beta, g_prox=g_prox_stochastic, g_blocks=groups, alpha=alpha)
        assert opt3.certificate < 1e-2
        np.testing.assert_allclose(opt2.x, opt3.x, atol=1e-1)

        opt4 = stochastic.fmin_PSSAGA_fast(
            stochastic.f_logistic, stochastic.deriv_logistic,
            X_sparse, y, np.zeros(n_features), step_size=step_size,
            gamma=beta, h_prox=g_prox_stochastic, h_blocks=groups, alpha=alpha)
        assert opt4.certificate < 1e-2
        np.testing.assert_allclose(opt3.x, opt4.x, atol=1e-1)


def test_fused_lasso():
    """Test sparse problems with group structure
    """

    alpha = 1.0

    def g_prox_stochastic(step_size, x, low, high):
        a = x[low] - x[low + 1]
        z = np.fmax(a - 2 * step_size, 0) - np.fmax(- a - 2 * step_size, 0) - a
        x[low] += z / 2.
        x[low + 1] -= z / 2.

    g_groups = np.arange(n_features) // 2
    h_groups = np.arange(1, n_features + 1) // 2

    for X in (X_sparse,):
        step_size = stochastic.compute_step_size('logistic', X, alpha)
        def loss(x):
            return logistic._logistic_loss(x, X, y, alpha * n_samples) / n_samples

        def grad(x):
            return logistic._logistic_loss_and_grad(x, X, y, alpha * n_samples)[1] / n_samples

        for beta in np.logspace(-3, 3, 3):

            opt = fmin_PGD(
                loss, grad, prox.prox_tv1d, np.zeros(n_features),
                step_size=step_size, alpha=beta)

            # PSSAGA now!
            opt3 = stochastic.fmin_PSSAGA_fast(
                stochastic.f_logistic, stochastic.deriv_logistic,
                X, y, np.zeros(n_features), step_size=step_size,
                alpha=alpha, beta=beta, gamma=beta, g_prox=g_prox_stochastic,
                g_blocks=g_groups, h_prox=g_prox_stochastic, h_blocks=h_groups)
            np.testing.assert_allclose(opt.x, opt3.x, atol=1e-1)
