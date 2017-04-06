import itertools
import numpy as np
import copt as cp
import pytest

np.random.seed(0)
n_samples, n_features = 50, 20
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))

all_solvers = (
    ['PGD', cp.minimize_PGD, 1e-4],
    ['APGD', cp.minimize_APGD, 1e-4],
    ['DavisYin', cp.minimize_DavisYin, 1e-3],
    ['BCD', cp.minimize_BCD, 1e-2],
    ['SAGA', cp.minimize_SAGA, 1e-3])

loss_funcs = [cp.LogisticLoss, cp.SquaredLoss]
penalty_funcs = [cp.L1Norm]


@pytest.mark.parametrize("name_solver, solver, tol", all_solvers)
@pytest.mark.parametrize("loss", loss_funcs)
@pytest.mark.parametrize("penalty", penalty_funcs)
def test_optimize(name_solver, solver, tol, loss, penalty):
    for alpha, beta in itertools.product(
            np.logspace(-3, 3, 5), np.logspace(-3, 3, 5)):
        f = loss(X, y, alpha)
        g = penalty(beta)
        ss = 1. / f.lipschitz_constant()
        opt = solver(f, g)
        gmap = (opt.x - g.prox(opt.x - ss * f.gradient(opt.x), ss)) / ss
        assert np.linalg.norm(gmap) < tol, name_solver


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
