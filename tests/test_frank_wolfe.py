import numpy as np
import copt as cp
import pytest

np.random.seed(0)
n_samples, n_features = 20, 16
A = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# we will use a logistic loss, which can't have values
# greater than 1
b = np.abs(b / np.max(np.abs(b)))

loss_funcs = [
    cp.utils.LogLoss,
    cp.utils.SquareLoss,
    ]


@pytest.mark.parametrize("loss_grad", loss_funcs)
def test_FW_L1(loss_grad):
    f_grad = loss_grad(A, b, 1./n_samples).f_grad
    alpha = 1.
    l1ball = cp.utils.L1Ball(alpha)
    opt = cp.minimize_FW(
        f_grad, l1ball.lmo, np.zeros(n_features), tol=0,
        max_iter=5000)
    assert np.isfinite(opt.x).sum() == n_features

    L = cp.utils.get_lipschitz(A, 'square', 1./n_samples)
    ss = 1/L
    grad = f_grad(opt.x)[1]
    grad_map = (opt.x - l1ball.prox(opt.x - ss*grad, ss))/ss
    assert np.linalg.norm(grad_map) < 0.015


@pytest.mark.parametrize("obj", loss_funcs)
def test_FW_trace(obj):
    f = obj(A, b, 1./n_samples)
    alpha = 1.
    L = f.lipschitz()
    traceball = cp.utils.TraceBall(alpha, (4, 4))
    opt = cp.minimize_FW(
        f.f_grad, traceball.lmo, np.zeros(n_features), tol=0,
        max_iter=5000, L=L)
    assert np.isfinite(opt.x).sum() == n_features

    ss = 1/L
    grad = f.f_grad(opt.x)[1]
    grad_map = (opt.x - traceball.prox(opt.x - ss*grad, ss))/ss
    assert np.linalg.norm(grad_map) < 1e-2


@pytest.mark.parametrize("obj", loss_funcs)
@pytest.mark.parametrize("backtracking", [True, False])
def test_PairwiseFW(obj, backtracking):
    f = obj(A, b, 1./n_samples)
    L = f.lipschitz()

    alpha = 1
    l1ball = cp.utils.L1Ball(alpha)
    opt = cp.minimize_PFW_L1(
        f.f_grad, alpha, n_features, tol=0,
        max_iter=5000, backtracking=backtracking, L=L)
    assert np.isfinite(opt.x).sum() == n_features

    ss = 1/L
    grad = f.f_grad(opt.x)[1]
    grad_map = (opt.x - l1ball.prox(opt.x - ss*grad, ss))/ss

    assert np.linalg.norm(grad_map) < 1e-10
