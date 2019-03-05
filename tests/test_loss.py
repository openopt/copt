import numpy as np
import copt as cp
from scipy import optimize
from scipy import sparse
from scipy import special

n_samples, n_features = 100, 10
A_dense = np.random.randn(n_samples, n_features)
b = np.random.uniform(0, 1, size=n_samples)
A_sparse = sparse.rand(n_samples, n_features, density=0.5, format='csr')


def test_loss_grad():
    for A in (A_dense, A_sparse):
        for loss in [
                cp.utils.LogLoss, cp.utils.SquareLoss, cp.utils.HuberLoss]:
            f = loss(A, b)
            err = optimize.check_grad(
                f, lambda x: f.f_grad(x)[1], np.random.randn(n_features))
            assert err < 1e-6


def test_log_hess():
    for A in (A_dense, A_sparse):
        f = cp.utils.LogLoss(A, b)
        x = np.random.randn(n_features)
        Hs = f.Hessian(x)

        def obj(x):
            return f.f_grad(x)[1][0]

        def grad(x):
            return f.Hessian(x)(np.eye(x.size)[0])
    
        err = optimize.check_grad(
                obj, grad, np.random.randn(n_features))
        assert err < 1e-6