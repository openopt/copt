import numpy as np
from scipy import sparse
import copt as cp
from copt import randomized
import pytest

np.random.seed(0)
n_samples, n_features = 20, 10
density = 0.5
A = sparse.random(n_samples, n_features, density=density)
A2 = sparse.random(n_samples, n_features+1, density=density)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# we will use a logistic loss, which can't have values
# greater than 1
b = np.abs(b / np.max(np.abs(b)))

all_solvers_unconstrained = (
    ['SAGA', cp.minimize_SAGA, 1e-3],
    ['SVRG', cp.minimize_SVRG, 1e-3],
    ['VRTOS', cp.minimize_VRTOS, 1e-3],
)


@pytest.mark.parametrize("name_solver, solver, tol", all_solvers_unconstrained)
def test_optimize(name_solver, solver, tol):
    f = cp.utils.LogLoss(A, b)
    for alpha in np.logspace(-3, 3, 3):
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density
        opt = solver(
            f.partial_deriv, A, b, np.zeros(n_features),
            1/(3 * L), alpha=alpha, max_iter=200, tol=1e-10)
        grad = cp.utils.LogLoss(A, b, alpha).f_grad(opt.x)[1]
        assert np.linalg.norm(grad) < tol, name_solver


def test_saga_l1():
    alpha = 1./n_samples
    f = cp.utils.LogLoss(A, b, alpha)
    for beta in np.logspace(-3, 3, 3):
        pen = cp.utils.L1Norm(beta)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        for solver in [cp.minimize_SAGA, cp.minimize_SVRG]:
            opt = solver(
                f.partial_deriv, A, b, np.zeros(n_features),
                1/(3 * L), alpha=alpha, max_iter=500, tol=1e-8,
                prox=pen.prox_factory(n_features))
            grad = cp.utils.LogLoss(A, b, alpha).f_grad(opt.x)[1]
            x = opt.x
            ss = 1./L
            # check that the gradient mapping vanishes
            grad_map = (x - pen.prox(x - ss*grad, ss))/ss
            assert np.linalg.norm(grad_map) < 1e-6


def test_vrtos():
    """Test VRTOS with no penalty."""
    alpha = 1./n_samples
    f = cp.utils.LogLoss(A, b)
    for beta in np.logspace(-3, 3, 3):
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        opt = cp.minimize_VRTOS(
            f.partial_deriv, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200)

        grad = cp.utils.LogLoss(A, b, alpha).f_grad(opt.x)[1]
        assert np.linalg.norm(grad) < 1e-6


def test_vrtos_l1():
    alpha = 1./n_samples
    f = cp.utils.LogLoss(A, b, alpha)
    for beta in np.logspace(-3, 3, 3):
        p_1 = cp.utils.L1Norm(beta)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        # blocks = np.arange(n_features)
        opt_1 = cp.minimize_VRTOS(
            f.partial_deriv, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_1=p_1.prox_factory(n_features))

        opt_2 = cp.minimize_VRTOS(
            f.partial_deriv, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_2=p_1.prox_factory(n_features))

        for x in [opt_1.x, opt_2.x]:
            full_prox = cp.utils.L1Norm(beta)
            grad = f.f_grad(x)[1]
            ss = 1./L
            # check that the gradient mapping vanishes
            grad_map = (x - full_prox.prox(x - ss*grad, ss))/ss
            assert np.linalg.norm(grad_map) < 1e-6


all_groups = [
    [np.arange(5)],
    np.arange(5).reshape((-1, 1)),
    [np.arange(5), [5], [6], [7], [8], [9]],
    [np.arange(5), np.arange(5, 10)]]


@pytest.mark.parametrize("groups", all_groups)
def test_gl(groups):
    alpha = 1./n_samples
    f = cp.utils.LogLoss(A, b, alpha)
    for beta in np.logspace(-3, 3, 3):
        p_1 = cp.utils.GroupL1(beta, groups)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        opt_1 = cp.minimize_VRTOS(
            f.partial_deriv, A, b, np.zeros(n_features),
            1/(3 * L), alpha=alpha, max_iter=200,
            prox_1=p_1.prox_factory(n_features))

        opt_2 = cp.minimize_VRTOS(
            f.partial_deriv, A, b, np.zeros(n_features),
            1/(3 * L), alpha=alpha, max_iter=200,
            prox_2=p_1.prox_factory(n_features))

        opt_3 = cp.minimize_SAGA(
            f.partial_deriv, A, b, np.zeros(n_features),
            1/(3 * L), alpha=alpha, max_iter=200,
            prox=p_1.prox_factory(n_features))

        for x in [opt_1.x, opt_2.x, opt_3.x]:
            grad = cp.utils.LogLoss(A, b, alpha).f_grad(x)[1]
            ss = 1./L
            # check that the gradient mapping vanishes
            grad_map = (x - p_1.prox(x - ss*grad, ss))/ss
            assert np.linalg.norm(grad_map) < 1e-6


def test_vrtos_ogl():
    """Test on overlapping group lasso"""
    alpha = 1./n_samples
    groups_1 = [np.arange(8)]
    groups_2 = [np.arange(5, 10)]
    f = cp.utils.LogLoss(A, b, alpha)
    for beta in np.logspace(-3, 3, 3):
        p_1 = cp.utils.GroupL1(beta, groups_1)
        p_2 = cp.utils.GroupL1(beta, groups_2)
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density

        opt_vrtos = cp.minimize_VRTOS(
            f.partial_deriv, A, b, np.zeros(n_features), 1/(3 * L),
            alpha=alpha, max_iter=200, prox_1=p_1.prox_factory(n_features),
            prox_2=p_2.prox_factory(n_features))

        opt_tos = cp.minimize_TOS(
            f.f_grad, np.zeros(n_features),
            prox_1=p_1.prox, prox_2=p_2.prox)

        norm = np.linalg.norm(opt_tos.x)
        if norm < 1e-10:
            norm = 1
        assert np.linalg.norm(opt_vrtos.x - opt_tos.x)/norm < 1e-4


@pytest.mark.parametrize("A_data", [A, A2])
def test_vrtos_fl(A_data):
    """Test on overlapping group lasso"""
    n_samples, n_features = A_data.shape
    alpha = 1./n_samples
    f = cp.utils.LogLoss(A_data, b, alpha)
    for beta in np.logspace(-3, 3, 3):
        pen = cp.utils.FusedLasso(beta)
        L = cp.utils.get_max_lipschitz(A_data, 'logloss') + alpha/density

        opt_vrtos = cp.minimize_VRTOS(
            f.partial_deriv, A_data, b, np.zeros(n_features),
            1/(3 * L), alpha=alpha, max_iter=2000,
            prox_1=pen.prox_1_factory(n_features),
            prox_2=pen.prox_2_factory(n_features), tol=0)

        opt_pgd = cp.minimize_PGD(
            f.f_grad, np.zeros(n_features),
            prox=pen.prox, max_iter=2000, tol=0)

        norm = np.linalg.norm(opt_pgd.x)
        if norm < 1e-10:
            norm = 1
        assert np.linalg.norm(opt_vrtos.x - opt_pgd.x)/norm < 1e-4

        # check also the gradient mapping
        ss = 1./L
        grad = f.f_grad(opt_vrtos.x)[1]
        grad_map = (opt_vrtos.x - pen.prox(opt_vrtos.x - ss*grad, ss))/ss
        assert np.linalg.norm(grad_map) < 1e-6
