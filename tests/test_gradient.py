import numpy as np
from scipy import optimize
from sklearn.linear_model import logistic
import copt as cp
import pytest

np.random.seed(0)
n_samples, n_features = 50, 20
A = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
b = np.sign(A.dot(w) + np.random.randn(n_samples))

all_solvers = (
    ['PGD', cp.minimize_PGD, 1e-3],
    ['PDHG', cp.minimize_PDHG, 1e-3],
    # ['APGD', cp.minimize_APGD, 1e-4],
    # ['DavisYin', cp.minimize_DavisYin, 1e-2],
    # ['BCD', cp.minimize_BCD, 1e-2],
    # ['SAGA', cp.minimize_SAGA, 1e-2]
)

loss_funcs = [cp.utils.LogLoss, cp.utils.SquareLoss]
penalty_funcs = [None]


def test_gradient():
    A = np.random.randn(10, 10)
    b = np.sign(np.random.randn(10))
    for f_grad in (
            cp.utils.logloss(A, b),
            cp.utils.SquareLoss(A, b).func_grad):
        f = lambda x: f_grad(x)[0]
        grad = lambda x: f_grad(x)[1]
        eps = optimize.check_grad(f, grad, np.random.randn(10))
        assert eps < 0.01


@pytest.mark.parametrize("name_solver, solver, tol", all_solvers)
@pytest.mark.parametrize("loss_grad", loss_funcs)
@pytest.mark.parametrize("penalty", penalty_funcs)
def test_optimize(name_solver, solver, tol, loss_grad, penalty):
    for alpha, beta in zip(
            np.logspace(-3, 3, 5), np.logspace(-3, 3, 5)):
        f_grad = loss_grad(A, b, alpha)
        opt = solver(f_grad, np.zeros(n_features))
        assert opt.certificate < tol, name_solver

#
# def test_optimize():
#
#     def logloss(x):
#         return logistic._logistic_loss(x, X, y, 1.)
#
#     def fprime_logloss(x):
#         return logistic._logistic_loss_and_grad(x, X, y, 1.)
#
#     L = np.eye(n_features)
#     opt = cp.minimize_PDHG(
#         fprime_logloss, None, None, L, np.zeros(n_features))
#     assert opt.success
#     sol_scipy = optimize.fmin_l_bfgs_b(
#         logloss, np.zeros(n_features),
#         fprime=lambda x: fprime_logloss(x)[1])[0]
#     np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)
