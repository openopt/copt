import itertools
import numpy as np
import copt as cp

np.random.seed(0)
n_samples, n_features = 100, 20
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))

all_solvers = (cp.minimize_PGD, cp.minimize_APGD,
               cp.minimize_DavisYin, cp.minimize_BCD,
               cp.minimize_SAGA)


def test_optimize():
    for alpha, beta, loss, solver in itertools.product(
            np.logspace(-3, 3, 5), np.logspace(-3, 3, 5),
            (cp.LogisticLoss, cp.SquaredLoss), all_solvers):
        f = loss(X, y, alpha)
        g = cp.L1Norm(beta)
        ss = 1. / f.lipschitz_constant()
        opt = solver(f, g)
        gmap = (opt.x - g.prox(opt.x - ss * f.gradient(opt.x), ss)) / ss
        assert np.linalg.norm(gmap) < 1e-3


#
#
# def test_fused():
#     """Solve the a problem with a fused lasso penalty in two
#     different, but equivalent ways:
#     1. Using a proximal gradient descent penalty and
#     the proximal operator of the fused lasso penalty.
#     2. Decomposing the fused lasso penalty in a sum of two
#     proximable penalties and using the three operator splitting.
#     """
#
#     def g_prox(step_size, x):
#         n_rows = x.size // 2
#         Lx = np.empty(n_rows)
#         for i in range(n_rows):
#             Lx[i] = x[2 * i] - x[2 * i + 1]
#         z = tv_prox.prox_L1(2 * step_size, Lx) - Lx
#         tmp = np.zeros(x.size)
#         for i in range(n_rows):
#             tmp[2 * i] = z[i]
#             tmp[2 * i + 1] = - z[i]
#         return x + tmp / 2
#
#     def h_prox(step_size, x):
#         tmp = x.copy()
#         tmp[1:] = g_prox(step_size, x[1:])
#         return tmp
#
#     for alpha in np.logspace(-3, 3, 5):
#         x0 = np.zeros(n_features)
#         opt1 = minimize_DavisYin(
#             logloss, fprime_logloss, h_prox, g_prox, x0.copy(),
#             alpha=alpha, beta=alpha)
#         # opt2 = fmin_PGD(
#         #     logloss, fprime_logloss, tv_prox.prox_tv1d, x0.copy(),
#         #     alpha=alpha)
#         #
#         # np.testing.assert_almost_equal(opt1.x, opt2.x)
#
