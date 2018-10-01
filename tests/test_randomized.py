import numpy as np
from scipy import sparse
import copt as cp
from copt import randomized
import pytest

np.random.seed(0)
n_samples, n_features = 20, 10
density = 0.5
A = sparse.random(n_samples, n_features, density=density)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# we will use a logistic loss, which can't have values
# greater than 1
b = np.abs(b / np.max(np.abs(b)))

all_solvers_unconstrained = (
    ['SAGA', cp.minimize_SAGA_L1, 1e-3],
    ['SVRG', cp.minimize_SVRG_L1, 1e-3],
    ['VRTOS', cp.minimize_VRTOS, 1e-3],
)


@pytest.mark.parametrize("name_solver, solver, tol", all_solvers_unconstrained)
def test_optimize(name_solver, solver, tol):
    for alpha in np.logspace(-3, 3, 3):
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density
        opt = solver(
            randomized.deriv_logistic, A, b, np.zeros(n_features),
            1/(3 * L), alpha=alpha, max_iter=200, tol=1e-10)
        grad = cp.utils.LogLoss(A, b, alpha).f_grad(opt.x)[1]
        assert np.linalg.norm(grad) < tol, name_solver


def saga_l1():
    alpha = 1./n_samples
    for beta in np.logspace(-3, 3, 3):
        full_l1 = cp.utils.L1Norm(beta)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        for solver in [cp.minimize_SAGA_L1, cp.minimize_SVRG_L1]:
            opt = solver(
                randomized.deriv_logistic, A, b, np.zeros(n_features),
                1/(3 * L), alpha=alpha, max_iter=500, tol=1e-8, beta=beta)
            grad = cp.utils.LogLoss(A, b, alpha).f_grad(opt.x)[1]
            x = opt.x
            ss = 1./L
            # check that the gradient mapping vanishes
            grad_map = (x - full_l1.prox(x - ss*grad, ss))/ss
            assert np.linalg.norm(grad_map) < 1e-6


def test_vrtos():
    """Test VRTOS with no penalty"""
    alpha = 1./n_samples
    for beta in np.logspace(-3, 3, 3):
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        opt = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200)

        grad = cp.utils.LogLoss(A, b, alpha).f_grad(opt.x)[1]
        assert np.linalg.norm(grad) < 1e-6


def test_vrtos_l1():
    alpha = 1./n_samples
    for beta in np.logspace(-3, 3, 3):
        p_1 = cp.utils.L1Norm(beta)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        # blocks = np.arange(n_features)
        opt_1 = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_1=p_1.prox_factory(n_features))

        opt_2 = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_2=p_1.prox_factory(n_features))

        for x in [opt_1.x, opt_2.x]:
            full_prox = cp.utils.L1Norm(beta)
            grad = cp.utils.LogLoss(A, b, alpha).f_grad(x)[1]
            ss = 1./L
            # check that the gradient mapping vanishes
            grad_map = (x - full_prox.prox(x - ss*grad, ss))/ss
            assert np.linalg.norm(grad_map) < 1e-6


def test_vrtos_gl():
    alpha = 1./n_samples
    groups_2 = [np.arange(5), np.arange(5, 10)]
    groups_1 = [np.arange(5)]
    for groups in [groups_1, groups_2]:
        for beta in np.logspace(-3, 3, 3):
            p_1 = cp.utils.GroupL1(beta, groups)
            L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

            opt_1 = cp.minimize_VRTOS(
                randomized.deriv_logistic, A, b, np.zeros(n_features),
                1/(3 * L), alpha=alpha, max_iter=200,
                prox_1=p_1.prox_factory(n_features))

            opt_2 = cp.minimize_VRTOS(
                randomized.deriv_logistic, A, b, np.zeros(n_features),
                1/(3 * L), alpha=alpha, max_iter=200,
                prox_2=p_1.prox_factory(n_features))

            for x in [opt_1.x, opt_2.x]:
                full_prox = cp.utils.GroupL1(beta, groups)
                grad = cp.utils.LogLoss(A, b, alpha).f_grad(x)[1]
                ss = 1./L
                # check that the gradient mapping vanishes
                grad_map = (x - full_prox.prox(x - ss*grad, ss))/ss
                assert np.linalg.norm(grad_map) < 1e-6


def test_vrtos_ogl():
    """Test on overlapping group lasso"""
    alpha = 1./n_samples
    groups_1 = [np.arange(8)]
    groups_2 = [np.arange(5, 10)]
    for beta in np.logspace(-3, 3, 3):
        p_1 = cp.utils.GroupL1(beta, groups_1)
        p_2 = cp.utils.GroupL1(beta, groups_2)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        opt_vrtos = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_1=p_1.prox_factory(n_features),
            prox_2=p_2.prox_factory(n_features))

        f_grad = cp.utils.LogLoss(A, b, alpha).f_grad
        opt_tos = cp.minimize_TOS(
            f_grad, np.zeros(n_features),
            g_prox=p_1.prox, h_prox=p_2.prox)

        norm = np.linalg.norm(opt_tos.x)
        if norm == 0:
            norm = 1
        assert np.linalg.norm(opt_vrtos.x - opt_tos.x)/norm < 1e-4
