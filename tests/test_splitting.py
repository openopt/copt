"""Tests for gradient-based methods."""
import copt as cp
from scipy import sparse
import numpy as np
import pytest

import copt.loss
import copt.penalty

np.random.seed(0)
n_samples, n_features = 20, 10
A = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# we will use a logistic loss, which can't have values
# greater than 1
b = np.abs(b / np.max(np.abs(b)))

all_solvers = (
    ["TOS", cp.minimize_three_split, 1e-12],
    ["PDHG", cp.minimize_primal_dual, 1e-5],
)

loss_funcs = [copt.loss.LogLoss, copt.loss.SquareLoss, copt.loss.HuberLoss]
penalty_funcs = [(None, None), (copt.penalty.L1Norm, None), (None, copt.penalty.L1Norm)]


def _get_prox(penalty):
    if penalty is not None:
        prox = penalty(1e-3).prox
    else:
        prox = None
    return prox


@pytest.mark.parametrize("name_solver, solver, tol", all_solvers)
@pytest.mark.parametrize("loss", loss_funcs)
@pytest.mark.parametrize("penalty", penalty_funcs)
def test_primal_dual_certificate(name_solver, solver, tol, loss, penalty):
    """Test a method on both the backtracking and fixed step size strategy."""
    max_iter = 1000
    for alpha in np.logspace(-1, 3, 3):
        obj = loss(A, b, alpha)
        prox_1 = _get_prox(penalty[0])
        prox_2 = _get_prox(penalty[1])
        trace = cp.utils.Trace(obj)
        opt = solver(
            obj.f_grad,
            np.zeros(n_features),
            prox_1=prox_1,
            prox_2=prox_2,
            tol=1e-12,
            max_iter=max_iter,
            callback=trace,
        )
        assert opt.certificate < tol, name_solver

        opt_2 = solver(
            obj.f_grad,
            np.zeros(n_features),
            prox_1=prox_1,
            prox_2=prox_2,
            max_iter=max_iter,
            tol=1e-12,
            line_search=False,
            step_size=1.0 / obj.lipschitz,
        )
        assert opt.certificate < tol, name_solver
        assert opt_2.certificate < tol, name_solver


@pytest.mark.parametrize("line_search", [False, True])
def test_PDHG_Lasso(line_search):
    # test the PDHG on a 1d-TV problem where we also
    loss = copt.loss.SquareLoss(A, b)
    alpha = 0.1
    L = np.eye(A.shape[1])  # (np.diag(np.ones(A.shape[1]), k=0))[:-1]
    opt1 = copt.minimize_primal_dual(
        loss.f_grad,
        np.zeros(n_features),
        prox_1=None,
        prox_2=copt.penalty.L1Norm(alpha).prox,
        L=L,
        tol=1e-14,
        line_search=line_search,
        step_size=0.4,
    )

    opt2 = copt.minimize_proximal_gradient(
        loss.f_grad,
        np.zeros(n_features),
        prox=copt.penalty.L1Norm(alpha).prox,
        tol=1e-12,
    )

    assert np.linalg.norm(opt1.x - opt2.x) / np.linalg.norm(opt1.x) < 1e-3


@pytest.mark.parametrize("line_search", [False, True])
def test_PDHG_FusedLasso(line_search):
    # test the PDHG on a 1d-TV problem where we also
    loss = copt.loss.SquareLoss(A, b)
    alpha = 0.1
    L = (np.diag(np.ones(A.shape[1]), k=0) - np.diag(np.ones(A.shape[1] - 1), k=1))[:-1]
    opt1 = copt.minimize_primal_dual(
        loss.f_grad,
        np.zeros(n_features),
        prox_1=None,
        prox_2=copt.penalty.L1Norm(alpha).prox,
        L=L,
        tol=1e-14,
        line_search=line_search,
        step_size=0.4,
    )

    opt2 = copt.minimize_proximal_gradient(
        loss.f_grad,
        np.zeros(n_features),
        prox=copt.penalty.FusedLasso(alpha).prox,
        tol=1e-12,
    )

    assert np.linalg.norm(opt1.x - opt2.x) / np.linalg.norm(opt1.x) < 1e-3


@pytest.mark.parametrize("regularization", np.logspace(-5, 1, 4))
@pytest.mark.parametrize("line_search", [False, True])
def test_PDHG_TV2D(regularization, line_search):
    # test the PDHG on a 2d-TV problem where we also

    img = np.random.randn(10, 10)
    n_rows, n_cols = img.shape
    n_features = n_rows * n_cols
    loss = copt.loss.SquareLoss(np.eye(n_features), img.ravel())

    def g_prox(x, gamma, pen=regularization):
        return cp.tv_prox.prox_tv1d_cols(gamma * pen, x, n_rows, n_cols)

    def h_prox(x, gamma, pen=regularization):
        return cp.tv_prox.prox_tv1d_rows(gamma * pen, x, n_rows, n_cols)

    opt1 = copt.minimize_primal_dual(
        loss.f_grad,
        np.zeros(n_features),
        prox_1=g_prox,
        prox_2=h_prox,
        tol=1e-14,
        line_search=line_search,
        #step_size=0.4,
    )

    opt2 = copt.minimize_three_split(
        loss.f_grad,
        np.zeros(n_features),
        prox_1=g_prox,
        prox_2=h_prox,
        tol=1e-12,
    )

    assert np.linalg.norm(opt1.x - opt2.x) / np.linalg.norm(opt1.x) < 1e-2
