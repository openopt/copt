"""Tests for the Stochastic Frank-Wolfe algorithm."""
import numpy as np
import pytest
from scipy import optimize, sparse
import copt as cp

np.random.seed(0)
n_samples, n_features = 20, 16
A = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# we will use a logistic loss, which can't have values
# greater than 1
b = np.abs(b / np.max(np.abs(b)))

LOSS_FUNCS = [cp.utils.LogLoss]

def test_fw_api():
    """Check that FW takes the right arguments and raises the right exceptions."""

    # test that the algorithm does not fail if x0
    # is a tuple
    f = cp.utils.LogLoss(A, b, 1.0 / n_samples)
    cb = cp.utils.Trace(f)
    alpha = 1.0
    l1ball = cp.utils.L1Ball(alpha)
    cp.randomized.minimize_sfw(
        f.partial_deriv,
        A,
        b,
        [0] * n_features,
        l1ball.lmo,
        tol=0,
        callback=cb,
        )


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("loss_grad", LOSS_FUNCS)
def test_sfw_l1(loss_grad, alpha):
    """Test result of FW algorithm with L1 constraint."""
    f = loss_grad(A, b, 1.0 / n_samples)
    cb = cp.utils.Trace(f)
    l1ball = cp.utils.L1Ball(alpha)
    opt = cp.randomized.minimize_sfw(
        f.partial_deriv,
        A,
        b,
        np.zeros(n_features),
        l1ball.lmo,
        tol=1e-3,
        callback=cb,
    )
    # assert np.isfinite(opt.x).sum() == n_features


@pytest.mark.parametrize("A", [sparse.random(n_samples, n_features, 0.1,
                                             fmt)
                               for fmt in ['coo', 'csr', 'csc', 'lil']])
def test_sfw_sparse(A):
    """Check that SFW runs on sparse data matrices and initial values."""

    f = cp.utils.LogLoss(A, b, 1.0 / n_samples)
    cb = cp.utils.Trace(f)
    alpha = 1.0
    l1ball = cp.utils.L1Ball(alpha)
    cp.randomized.minimize_sfw(
        f.partial_deriv,
        A,
        b,
        sparse.csr_matrix((n_features, 1)),
        l1ball.lmo,
        tol=0,
        callback=cb,
        )

