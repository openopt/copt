import numpy as np
from scipy import optimize, sparse
from sklearn.linear_model import logistic
from copt import proximal_gradient
from copt import primal_dual
from copt.prox_tv import prox_tv1d

np.random.seed(0)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():

    def logloss(x):
        return logistic._logistic_loss(x, X, y, 1.)

    def fprime_logloss(x):
        return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]

    L = np.eye(n_features)
    opt = primal_dual(
        fprime_logloss, None, None, L, np.zeros(n_features))
    assert opt.success
    sol_scipy = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features),
        fprime=fprime_logloss)[0]
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)


def test_lasso():
    for alpha in np.logspace(-3, 3, 3):

        def logloss(x):
            return logistic._logistic_loss(x, X, y, 0.)

        def fprime_logloss(x):
            return logistic._logistic_loss_and_grad(x, X, y, 0.)[1]

        def l1_prox(x, step_size):
            return np.fmax(x - step_size, 0) - np.fmax(-x - step_size, 0)

        L = np.eye(n_features)
        opt_proximal = proximal_gradient(
            logloss, fprime_logloss, l1_prox, np.zeros(n_features),
            tol=1e-24, alpha=alpha)
        opt_primal_dual = primal_dual(
            fprime_logloss, l1_prox, None, L, np.zeros(n_features),
            alpha=alpha)
        assert opt_primal_dual.success
        np.testing.assert_allclose(
            opt_proximal.x, opt_primal_dual.x, rtol=1e-1)

        # same thing but using the other operator
        opt_primal_dual2 = primal_dual(
            fprime_logloss, None, l1_prox, L, np.zeros(n_features),
            beta=alpha)
        np.testing.assert_allclose(
            opt_proximal.x, opt_primal_dual2.x, atol=1e-3)


def test_fused():
    """
    Test that it can solve a problem with fused lasso regularization
    using only the L1-prox and an appropriate L linear operator.
    """
    for alpha in np.logspace(-3, 3, 3):

        def logloss(x):
            return logistic._logistic_loss(x, X, y, 0.)

        def fprime_logloss(x):
            return logistic._logistic_loss_and_grad(x, X, y, 0.)[1]

        def l1_prox(x, step_size):
            return np.fmax(x - step_size, 0) - np.fmax(-x - step_size, 0)

        L = sparse.diags([1, -1], [0, 1], shape=(n_features - 1, n_features))
        # solve the problem using the fused lasso proximal operator
        # (only for reference)
        opt_proximal = proximal_gradient(
            logloss, fprime_logloss, prox_tv1d, np.zeros(n_features),
            tol=1e-24, max_iter=10000, alpha=alpha)

        opt_primal_dual = primal_dual(
            fprime_logloss, None, l1_prox, L, np.zeros(n_features),
            beta=alpha, verbose=True, step_size_y=1)
        assert opt_primal_dual.success
        np.testing.assert_allclose(
            opt_proximal.x, opt_primal_dual.x, atol=1e-1)

