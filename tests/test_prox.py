import numpy as np
import copt as cp
from copt import tv_prox
from numpy import testing


def test_tv1_prox():
    """
    Use the properties of strongly convex functions to test the implementation
    of the TV1D proximal operator. In particular, we use the following inequality
    applied to the proximal objective function: if f is mu-strongly convex then

          f(x) - f(x^*) >= ||x - x^*||^2 / (2 mu)

    where x^* is the optimum of f.
    """
    n_features = 10
    gamma = np.random.rand()
    epsilon = 1e-10  # account for some numerical errors

    tv_norm = lambda x: np.sum(np.abs(np.diff(x)))
    for _ in range(1000):
        x = np.random.randn(n_features)
        x_next = tv_prox.prox_tv1d(gamma, x)
        diff_obj = tv_norm(x) - tv_norm(x_next)
        testing.assert_array_less(
        ((x - x_next) ** 2).sum() / gamma, (1 + epsilon) * diff_obj)


def test_tv2_prox():
    """
    similar test, but for 2D total variation penalty.
    """
    np.random.seed(0)
    n_rows, n_cols = 6, 8
    n_features = n_rows * n_cols
    gamma = np.random.rand()
    epsilon = 0.1  # account for some numerical errors

    def tv_norm(x, n_rows, n_cols):
        X = x.reshape((n_rows, n_cols))
        return np.sum(np.abs(np.diff(X, 0))) + np.sum(np.abs(np.diff(X, 1)))

    for nrun in range(20):
        x = np.random.randn(n_features)
        x_next = tv_prox.prox_tv2d(gamma, x, n_rows, n_cols, tol=1e-10, max_iter=10000)
        diff_obj = tv_norm(x, n_rows, n_cols) - tv_norm(x_next, n_rows, n_cols)
        testing.assert_array_less(
            ((x - x_next) ** 2).sum() / gamma, (1 + epsilon) * diff_obj)


def test_tv2d_linear_operator():
    n_rows, n_cols = 20, 10
    def TV(w):
        img = w.reshape((n_rows, n_cols))
        tmp1 = np.abs(np.diff(img, axis=0))
        tmp2 = np.abs(np.diff(img, axis=1))
        return tmp1.sum() + tmp2.sum()

    L = tv_prox.tv2d_linear_operator(n_rows, n_cols)
    x = np.random.randn(n_rows * n_cols)
    testing.assert_almost_equal(
        np.abs(L.dot(x)).sum(), TV(x))


def test_three_inequality():
    """Test the L1 prox using the three point inequality

    The three-point inequality is described e.g., in Lemma 1.4
    in "Gradient-Based Algorithms with Applications to Signal
    Recovery Problems", Amir Beck and Marc Teboulle
    """
    n_features = 10

    l1 = cp.utils.L1Norm(1.)
    blocks = np.arange(n_features) // 2
    gl1 = cp.utils.GroupL1(1., blocks)

    for loss in [l1, gl1]:
        for _ in range(10):
            z = np.random.randn(n_features)
            u = np.random.randn(n_features)
            xi = loss.prox(z, 1.)

            lhs = 2 * (loss(xi) - loss(u))
            rhs = np.linalg.norm(u - z) ** 2 - \
                np.linalg.norm(u - xi) ** 2 - \
                np.linalg.norm(xi - z) ** 2
            assert lhs <= rhs
