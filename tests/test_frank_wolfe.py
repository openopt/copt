import numpy as np
from scipy import optimize
from sklearn.linear_model import logistic
import copt as cp
import pytest

np.random.seed(0)
n_samples, n_features = 50, 20
A = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# we will use a logistic loss, which can't have values
# greater than 1
b = np.abs(b / np.max(np.abs(b)))


all_solvers = (
    ['FW', cp.minimize_FW, 2e-2],
    # ['PFW', cp.minimize_PFW_L1, 1e-3],
)
loss_funcs = [
    cp.utils.LogLoss, 
    cp.utils.SquareLoss,
    ]



@pytest.mark.parametrize("name_solver, solver, tol", all_solvers)
@pytest.mark.parametrize("loss_grad", loss_funcs)
def test_optimize(name_solver, solver, tol, loss_grad):
    f_grad = loss_grad(A, b, 1./n_samples).f_grad
    alpha = 1.
    l1ball = cp.utils.L1Ball(alpha)
    opt = solver(
        f_grad, l1ball.lmo, np.zeros(n_features), tol=0,
        max_iter=5000)
    assert np.isfinite(opt.x).sum() == n_features

    L = cp.utils.get_lipschitz(A, 'square', 1./n_samples)
    ss = 1/L
    grad = f_grad(opt.x)[1]
    grad_map = (opt.x - l1ball.prox(opt.x - ss*grad, ss))/ss
    assert np.linalg.norm(grad_map) < tol, name_solver
