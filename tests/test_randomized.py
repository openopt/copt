import numpy as np
from scipy import optimize, sparse
from sklearn.linear_model import logistic
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
        opt = solver(randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L), alpha=alpha, max_iter=200, tol=1e-10)
        grad = cp.utils.LogLoss(A, b, alpha).f_grad(opt.x)[1]
        assert np.linalg.norm(grad) < tol, name_solver


def saga_l1():
    alpha = 1./n_samples
    for beta in np.logspace(-3, 3, 3):
        full_l1 = cp.utils.L1Norm(beta)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density
        p_1 = cp.randomized.prox_l1(beta)

        for solver in [cp.minimize_SAGA_L1, cp.minimize_SVRG_L1]:
            opt = solver(
                randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
                alpha=alpha, max_iter=500, tol=1e-8, beta=beta)
            grad = cp.utils.LogLoss(A, b, alpha).f_grad(opt.x)[1]
            x = opt.x
            ss = 1./L
            # check that the gradient mapping vanishes
            grad_map = (x - full_l1.prox(x - ss*grad, ss))/ss
            assert np.linalg.norm(grad_map) < 1e-6

def test_vrtos():
    alpha = 1./n_samples
    for beta in np.logspace(-3, 3, 3):
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        # giving it strange blocks because it should not change the solution
        blocks = np.arange(n_features, dtype=np.int) // 5
        # blocks = np.arange(n_features)
        opt_1 = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, blocks_1=blocks)

        opt_2 = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200)

        for x in [opt_1.x, opt_2.x]:
            grad = cp.utils.LogLoss(A, b, alpha).f_grad(x)[1]
            ss = 1./L
            # check that the gradient mapping vanishes
            assert np.linalg.norm(grad) < 1e-6


def test_vrtos_l1():
    alpha = 1./n_samples
    for beta in np.logspace(-3, 3, 3):
        p_1 = cp.randomized.prox_l1(beta)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        # giving him strange blocks because it should not change the solution
        blocks = np.zeros(n_features, dtype=np.int)
        # blocks = np.arange(n_features)
        opt_1 = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_1=p_1, blocks_1=blocks)

        opt_2 = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_2=p_1)

        for x in [opt_1.x, opt_2.x]:
            full_prox = cp.utils.L1Norm(beta)
            grad = cp.utils.LogLoss(A, b, alpha).f_grad(x)[1]
            ss = 1./L
            # check that the gradient mapping vanishes
            grad_map = (x - full_prox.prox(x - ss*grad, ss))/ss
            assert np.linalg.norm(grad_map) < 1e-6


def test_vrtos_gl():
    alpha = 1./n_samples
    blocks = np.arange(n_features)//2
    for beta in np.logspace(-3, 3, 3):
        p_1 = cp.randomized.prox_gl(beta)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        opt_1 = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_1=p_1, blocks_1=blocks)

        opt_2 = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_2=p_1, blocks_2=blocks)

        for x in [opt_1.x, opt_2.x]:
            full_prox = cp.utils.GroupL1(beta, blocks)
            grad = cp.utils.LogLoss(A, b, alpha).f_grad(x)[1]
            ss = 1./L
            # check that the gradient mapping vanishes
            grad_map = (x - full_prox.prox(x - ss*grad, ss))/ss
            assert np.linalg.norm(grad_map) < 1e-6


def test_vrtos_gl():
    alpha = 1./n_samples
    blocks = np.arange(n_features)//2
    for beta in np.logspace(-3, 3, 3):
        p_1 = cp.randomized.prox_gl(beta)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        opt_1 = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_1=p_1, blocks_1=blocks)

        opt_2 = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_2=p_1, blocks_2=blocks)

        for x in [opt_1.x, opt_2.x]:
            full_prox = cp.utils.GroupL1(beta, blocks)
            grad = cp.utils.LogLoss(A, b, alpha).f_grad(x)[1]
            ss = 1./L
            # check that the gradient mapping vanishes
            grad_map = (x - full_prox.prox(x - ss*grad, ss))/ss
            assert np.linalg.norm(grad_map) < 1e-6


def test_vrtos_ogl():
    """Test on overlapping group lasso"""
    alpha = 1./n_samples
    blocks_1 = np.arange(n_features)//2
    blocks_2 = np.arange(n_features)//3
    for beta in np.logspace(-3, 3, 3):
        p_1 = cp.randomized.prox_gl(beta)
        p_2 = cp.randomized.prox_gl(beta)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        opt_vrtos = cp.minimize_VRTOS(
            randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_1=p_1, blocks_1=blocks_1,
            prox_2=p_1, blocks_2=blocks_2)

        f_grad = cp.utils.LogLoss(A, b, alpha).f_grad
        group_1 = cp.utils.GroupL1(beta, blocks_1)
        group_2 = cp.utils.GroupL1(beta, blocks_2)
        opt_tos = cp.minimize_TOS(
            f_grad, np.zeros(n_features),
            g_prox=group_1.prox, h_prox=group_2.prox)

        norm = np.linalg.norm(opt_tos.x) 
        if norm == 0:
            norm = 1
        assert np.linalg.norm(opt_vrtos.x - opt_tos.x)/norm < 1e-4
