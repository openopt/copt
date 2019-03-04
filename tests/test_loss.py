import numpy as np
import copt as cp
from scipy import optimize
from scipy import sparse

n_samples, n_features = 10, 10
A = np.random.randn(n_samples, n_features)
b = np.random.uniform(0, 1, size=n_samples)
A_sparse = sparse.rand(n_samples, n_features, density=0.5, format='csr')


def test_log_grad():
    for A_data in (A, A_sparse):
        for loss in [
                cp.utils.LogLoss, cp.utils.SquareLoss, cp.utils.HuberLoss]:
            f = loss(A_data, b)
            err = optimize.check_grad(
                f, lambda x: f.f_grad(x)[1], np.random.randn(n_features))
            assert err < 1e-6
