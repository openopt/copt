import numpy as np
from gdprox import prox_tv1d, prox_tv2d


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
    epsilon = 1e-6  # account for some numerica errors

    tv_norm = lambda x: np.sum(np.abs(np.diff(x)))
    for _ in range(100):
        x = np.random.randn(n_features)
        x_next = prox_tv1d(x, gamma)
        diff_obj = tv_norm(x) - tv_norm(x_next)
        assert (1 + epsilon) * diff_obj >= ((x - x_next) ** 2).sum() / gamma


def test_tv2_prox():
    """
    similar test, but for 2D total variation penalty.
    """
    n_rows, n_cols = 6, 8
    n_features = n_rows * n_cols
    gamma = np.random.rand()

    def tv_norm(x, n_rows, n_cols):
        X = x.reshape((n_rows, n_cols))
        return np.sum(np.abs(np.diff(X, 0))) + np.sum(np.abs(np.diff(X, 1)))

    for nrun in range(200):
        x = np.random.randn(n_features)
        x2 = prox_tv2d(x, gamma, n_rows, n_cols)
        diff_obj = tv_norm(x, n_rows, n_cols) - tv_norm(x, n_rows, n_cols)
        assert diff_obj >= ((x - x2) ** 2).sum() / (2 * gamma)