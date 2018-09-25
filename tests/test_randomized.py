import numpy as np
from scipy import optimize, sparse
from sklearn.linear_model import logistic
import copt as cp
from copt import randomized
import pytest

np.random.seed(0)
n_samples, n_features = 50, 20
density = 0.5
A = sparse.random(n_samples, n_features, density=density)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# we will use a logistic loss, which can't have values
# greater than 1
b = np.abs(b / np.max(np.abs(b)))

all_solvers_unconstrained = (
    ['SAGA', cp.minimize_SAGA_L1, 1e-3],
    ['VRTOS', cp.minimize_VRTOS, 1e-3],
)



@pytest.mark.parametrize("name_solver, solver, tol", all_solvers_unconstrained)
def test_optimize(name_solver, solver, tol):
    for alpha in np.logspace(-3, 3, 3):
        L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density
        opt = solver(randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L), alpha=alpha, max_iter=200, tol=1e-10)
        grad = cp.utils.LogLoss(A, b, alpha).func_grad(opt.x)[1]
        assert np.linalg.norm(grad) < tol, name_solver


# def test_vrtos_l1():
#     alpha = 10./n_samples
#     for beta in np.logspace(-3, 3, 3):
#         full_l1 = cp.utils.L1Norm(beta)
#         L = cp.utils.get_max_lipschitz(A, 'logloss') + alpha/density
#         p_1 = cp.randomized.prox_l1(beta)
# 
# 
#         opt = cp.minimize_VRTOS(
#             randomized.deriv_logistic, A, b, np.zeros(n_features), 1/(3 * L),
#             alpha=alpha, max_iter=10000, tol=1e-10, prox_1=p_1,
#             blocks_1=np.arange(n_features))
#         grad = cp.utils.LogLoss(A, b, alpha).func_grad(opt.x)[1]
#         x = opt.x
#         ss = 1./L
#         # check that the gradient mapping vanishes
#         grad_map = (x - full_l1.prox(x - ss*grad, ss))/ss
#         assert np.linalg.norm(grad_map) < 1e-3
