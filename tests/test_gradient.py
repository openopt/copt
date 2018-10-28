"""
Tests for gradient-based methods
"""
import numpy as np
from scipy import optimize
import copt as cp
import pytest

np.random.seed(0)
n_samples, n_features = 20, 10
A = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# we will use a logistic loss, which can't have values
# greater than 1
b = np.abs(b / np.max(np.abs(b)))

all_solvers = (
    ['PGD', cp.minimize_PGD, 1e-6],
    # ['APGD', cp.minimize_APGD, 1e-3],
)

loss_funcs = [
    cp.utils.LogLoss, cp.utils.SquareLoss, cp.utils.HuberLoss]
penalty_funcs = [None, cp.utils.L1Norm]


def test_gradient():
    for _ in range(20):
        A = np.random.randn(10, 5)
        b = np.random.rand(10)
        for loss in loss_funcs:
            f_grad = loss(A, b).f_grad
            f = lambda x: f_grad(x)[0]
            grad = lambda x: f_grad(x)[1]
            eps = optimize.check_grad(f, grad, np.random.randn(5))
            assert eps < 0.001


@pytest.mark.parametrize("name_solver, solver, tol", all_solvers)
@pytest.mark.parametrize("loss", loss_funcs)
@pytest.mark.parametrize("penalty", penalty_funcs)
def test_optimize(name_solver, solver, tol, loss, penalty):
    """
    Test a method on both the backtracking and fixed step size strategy
    """
    for alpha in np.logspace(-1, 3, 3):
        obj = loss(A, b, alpha)
        prox = penalty(1e-3).prox
        opt = solver(
            obj.f_grad, np.zeros(n_features), prox=prox, tol=1e-12)
        certificate = np.linalg.norm(obj.f_grad(opt.x)[1])
        assert certificate < tol, name_solver

        opt_2 = solver(
            obj.f_grad, np.zeros(n_features), prox=prox, max_iter=5000,
            tol=1e-10, backtracking=False, step_size=1/obj.lipschitz)
        certificate = np.linalg.norm(obj.f_grad(opt_2.x)[1])
        assert certificate < tol, name_solver
