"""Tests for the Stochastic Frank-Wolfe algorithms."""
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
VARIANTS = ['SAGA', 'SAG', 'MK', 'LF']


@pytest.mark.parametrize("variant", VARIANTS)
def test_fw_api(variant):
    """Check that SFW algorithms take the right arguments and raises the right exceptions."""

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
        variant=variant
        )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("loss_grad", LOSS_FUNCS)
def test_sfw_l1(variant, loss_grad, alpha):
    """Test SFW algorithms with L1 constraint."""
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
        variant=variant
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("loss_grad", LOSS_FUNCS)
def test_sfw_gap_traceback(variant, loss_grad, alpha):
    """Test outputting the FW gap for SFW algorithms."""
    f = loss_grad(A, b, 1.0 / n_samples)
    l1ball = cp.utils.L1Ball(alpha)

    def fw_gap(x):
        _, grad = f.f_grad(x)
        return l1ball.lmo(-grad, x)[0].dot(-grad)

    class TraceGaps(cp.utils.Trace):
        def __init__(self, f=None, freq=1):
            super(TraceGaps, self).__init__(f, freq)
            self.trace_gaps = []

        def __call__(self, dl):
            self.trace_gaps.append(fw_gap(dl['x']))
            super(TraceGaps, self).__call__(dl)

    cb = TraceGaps(f)

    opt = cp.randomized.minimize_sfw(
        f.partial_deriv,
        A,
        b,
        np.zeros(n_features),
        l1ball.lmo,
        tol=1e-3,
        callback=cb,
        variant=variant
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("A", [sparse.random(n_samples, n_features, 0.1,
                                             fmt)
                               for fmt in ['coo', 'csr', 'csc', 'lil']])
def test_sfw_sparse(variant, A):
    """Check that SFW algorithms run on sparse data matrices and initial values."""

    f = cp.utils.LogLoss(A, b, 1.0 / n_samples)
    cb = cp.utils.Trace(f)
    alpha = 1.0
    l1ball = cp.utils.L1Ball(alpha)
    cp.randomized.minimize_sfw(
        f.partial_deriv,
        A,
        b,
        np.zeros(n_features),
        l1ball.lmo,
        tol=0,
        callback=cb,
        variant=variant
        )

