import numpy as np
from scipy import sparse
import copt as cp

np.random.seed(0)
n_samples, n_features = 100, 10
X_sparse = sparse.random(n_samples, n_features, density=0.2, format='csr')
X_dense = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():
    for alpha in np.logspace(-1, 3, 3):
        for X in (X_dense, X_sparse):
            f = cp.LogisticLoss(X, y, alpha)
            opt = cp.minimize_SAGA(f)
            assert np.linalg.norm(f.gradient(opt.x)) < 1e-3

#
# def test_prox_sparse():
#     alpha = 1.0 / n_samples
#
#     for X in (X_dense, X_sparse):
#         def loss(x):
#             return logistic._logistic_loss(x, X, y, 1.0) / n_samples
#
#         def grad(x):
#             return logistic._logistic_loss_and_grad(x, X, y, 1.0)[1] / n_samples
#
#         step_size = stochastic.compute_step_size('logistic', X, alpha * n_samples)
#         for beta in np.logspace(-3, 3, 3):
#             # opt = stochastic.fmin_SAGA(
#             #     stochastic.f_logistic, stochastic.deriv_logistic,
#             #     X, y, np.zeros(n_features), step_size=step_size,
#             #     alpha=alpha, beta=beta, g_prox=stochastic.prox_L1)
#             opt2 = fmin_PGD(
#                 loss, grad, prox.prox_L1, np.zeros(n_features),
#                 alpha=beta)
#             # # assert opt.success
#             # np.testing.assert_allclose(opt.x, opt2.x, atol=1e-1)
#
#             opt3 = stochastic.fmin_PSSAGA(
#                 stochastic.f_logistic, stochastic.deriv_logistic,
#                 X, y, np.zeros(n_features), step_size=step_size,
#                 alpha=alpha, gamma=beta, h_prox=stochastic.prox_L1, tol=0)
#
#             def g_prox(step_size, input, output, low, high, weights):
#                 for i in range(low, high):
#                     output[i] = input[i]
#                     stochastic.prox_L1(step_size * weights[i], output, i, i+1)
#
#             opt4 = stochastic.fmin_PSSAGA(
#                 stochastic.f_logistic, stochastic.deriv_logistic,
#                 X, y, np.zeros(n_features), step_size=step_size,
#                 alpha=alpha, beta=beta, g_prox=g_prox, tol=0)
#             np.testing.assert_allclose(opt2.x, opt3.x, rtol=1e-2)
#             assert np.abs(loss(opt2.x) - loss(opt3.x)) < 0.1
#             assert np.abs(loss(opt2.x) - loss(opt4.x)) < 0.1
#
#
# def test_prox_groups():
#     """Test sparse problems with group structure
#
#     The nonsmooth term that we use is
#         |x_1 - x_2| + |x_3 - x_4| + |x_5 - x_6|
#     """
#
#     alpha = 1.
#
#     def loss(x):
#         return logistic._logistic_loss(x, X_sparse, y, alpha * n_samples) / n_samples
#
#     def grad(x):
#         return logistic._logistic_loss_and_grad(x, X_sparse, y, alpha * n_samples)[1] / n_samples
#
#     prox_L1 = njit(prox.prox_L1)
#
#     def g_prox(step_size, y, low, high):
#         x = y[low:high]
#         n_rows = x.size // 2
#         Lx = np.empty(n_rows)
#         for i in range(n_rows):
#             Lx[i] = x[2 * i] - x[2 * i + 1]
#         z = prox_L1(2 * step_size, Lx) - Lx
#         tmp = np.zeros(x.size)
#         for i in range(n_rows):
#             tmp[2 * i] = z[i]
#             tmp[2 * i + 1] = - z[i]
#         y[low:high] = x + tmp / 2
#         return y.copy()
#
#     def g_prox_stochastic(step_size, x, low, high):
#         if high - low < 2:
#             return
#         a = x[low] - x[low + 1]
#         z = np.fmax(a - 2 * step_size, 0) - np.fmax(- a - 2 * step_size, 0) - a
#         x[low] += z / 2.
#         x[low + 1] -= z / 2.
#
#     groups = np.arange(n_features) // 2
#     step_size = stochastic.compute_step_size('logistic', X_sparse, alpha * n_samples)
#     for beta in np.logspace(-3, 3, 3):
#
#         opt = stochastic.fmin_SAGA(
#             stochastic.f_logistic, stochastic.deriv_logistic,
#             X_sparse, y, np.zeros(n_features), step_size=step_size,
#             beta=beta, g_prox=g_prox, g_blocks=groups, alpha=alpha, max_iter=10)
#         opt2 = fmin_PGD(
#             loss, grad, g_prox, np.zeros(n_features),
#             alpha=beta, g_prox_args=(0, n_features))
#         np.testing.assert_allclose(opt.x, opt2.x, atol=1e-1)
#
#         # PSSAGA now!
#         opt3 = stochastic.fmin_PSSAGA(
#             stochastic.f_logistic, stochastic.deriv_logistic,
#             X_sparse, y, np.zeros(n_features), step_size=step_size,
#             beta=beta, g_prox=g_prox_stochastic, g_blocks=groups, alpha=alpha)
#         assert opt3.certificate < 1e-2
#         np.testing.assert_allclose(opt2.x, opt3.x, atol=1e-1)
#
#         opt4 = stochastic.fmin_PSSAGA(
#             stochastic.f_logistic, stochastic.deriv_logistic,
#             X_sparse, y, np.zeros(n_features), step_size=step_size,
#             gamma=beta, h_prox=g_prox_stochastic, h_blocks=groups, alpha=alpha)
#         assert opt4.certificate < 1e-2
#         np.testing.assert_allclose(opt3.x, opt4.x, atol=1e-2)
#
#
# def test_fused_lasso():
#     """Test sparse problems with group structure
#     """
#
#     alpha = 1.0
#
#     @njit
#     def custom_prox(a, b, stepsize, d1, d2):
#         if a - stepsize / d1 >= b + stepsize / d2:
#             return a - stepsize / d1, b + stepsize / d2
#         elif b - stepsize / d2 >= a + stepsize / d1:
#             return a + stepsize / d1, b - stepsize / d2
#         else:
#             mean = (d1 * a + d2 * b) / (d1 + d2)
#             return mean, mean
#
#     def g_prox(step_size, x, z, low, high, weights):
#         if high - low == 1:
#             # this can be the first or the last feature
#             if low == 0:
#                 a, b = custom_prox(x[0], x[1], step_size, weights[0], weights[1])
#                 z[0] = a
#                 z[1] = b
#                 return
#             elif high == x.size:
#                 a, b = custom_prox(x[high - 2], x[high - 1], step_size, weights[high - 2], weights[high - 1])
#                 z[high - 1] = b
#                 z[high - 2] = a
#                 return
#         assert high - low == 2
#         a, b = custom_prox(x[low - 1], x[low], step_size, weights[low - 1], weights[low])
#         z[low] = b
#         z[low - 1] = a
#         a, b = custom_prox(x[low + 1], x[low + 2], step_size, weights[low + 1], weights[low + 2])
#         z[low + 1] = a
#         z[low + 2] = b
#
#     def h_prox(step_size, x, low, high):
#         if high - low < 2:
#             return
#         a = x[low] - x[low + 1]
#         z = np.fmax(a - 2 * step_size, 0) - np.fmax(- a - 2 * step_size, 0) - a
#         x[low] += z / 2.
#         x[low + 1] -= z / 2.
#
#     h_groups = np.arange(1, n_features + 1) // 2
#
#     for X in (X_sparse, X_dense):
#         step_size = stochastic.compute_step_size('logistic', X, alpha * n_samples)
#
#         def loss(x):
#             return logistic._logistic_loss(x, X, y, alpha * n_samples) / n_samples
#
#         def grad(x):
#             return logistic._logistic_loss_and_grad(x, X, y, alpha * n_samples)[1] / n_samples
#
#         for beta in np.logspace(-3, 3, 3):
#
#             opt = fmin_PGD(
#                 loss, grad, prox.prox_tv1d, np.zeros(n_features),
#                 step_size=step_size, alpha=beta, max_iter=10000, tol=0)
#
#             # PSSAGA now!
#             opt3 = stochastic.fmin_PSSAGA(
#                 stochastic.f_logistic, stochastic.deriv_logistic,
#                 X, y, np.zeros(n_features), step_size=step_size,
#                 alpha=alpha, beta=beta, gamma=beta, g_prox=g_prox,
#                 max_iter=10000, tol=0,
#                 h_prox=h_prox, h_blocks=h_groups)
#             np.testing.assert_allclose(opt.x, opt3.x, atol=1e-3)
