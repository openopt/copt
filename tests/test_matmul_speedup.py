import scipy.sparse as sparse
import numpy as np
import copt as cp

n_samples, n_features = 1000, 100
n_subset = 5
A_sparse = sparse.rand(n_samples, n_features, density=0.5, format="csr")
x = np.random.rand(n_features)
u = np.random.rand(n_subset)
idx = np.random.choice(n_samples, n_subset)


def test_fast_csr_vm():
    res = cp.utils.fast_csr_vm(u,
                               A_sparse.data, A_sparse.indptr, A_sparse.indices,
                               n_features, idx)
    assert np.allclose(res, cp.utils.safe_sparse_dot(u, A_sparse[idx]))


def test_fast_csr_mv():
    res = cp.utils.fast_csr_mv(A_sparse.data, A_sparse.indptr, A_sparse.indices,
                               x, idx)
    assert np.allclose(res, cp.utils.safe_sparse_dot(A_sparse[idx], x))
