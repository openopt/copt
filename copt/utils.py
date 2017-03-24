import numpy as np
from scipy import sparse
from numba import njit
from .tv_prox import prox_tv2d


class LogisticLoss:
    """Logistic regression loss function with L2 regularization

    This loss function is very popular for binary classification tasks.
    Labels (b) are assumed to be 1 or -1.
    """

    def __init__(self, A, b, alpha='auto'):
        self.b = b
        self.A = A
        self.n_features = self.A.shape[1]
        if alpha == 'auto':
            self.alpha = 1. / A.shape[0]
        else:
            self.alpha = alpha

    def __call__(self, x):
        # loss function to be optimized, it's the logistic loss
        z = self.A.dot(x)
        yz = self.b * z
        idx = yz > 0
        out = np.zeros_like(yz)
        out[idx] = np.log(1 + np.exp(-yz[idx]))
        out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
        out = out.mean() + .5 * self.alpha * x.dot(x)
        return out

    def gradient(self, x):
        def phi(t):
            # logistic function, returns 1 / (1 + exp(-t))
            idx = t > 0
            out = np.empty(t.size, dtype=np.float)
            out[idx] = 1. / (1 + np.exp(-t[idx]))
            exp_t = np.exp(t[~idx])
            out[~idx] = exp_t / (1. + exp_t)
            return out
        z = self.A.dot(x)
        z = phi(self.b * z)
        z0 = (z - 1) * self.b
        grad_w = self.A.T.dot(z0) / self.A.shape[0] + self.alpha * x
        return grad_w

    @staticmethod
    def partial_gradient_factory():
        @njit
        def partial_gradient(p, b):
            # compute p
            p *= b
            if p > 0:
                phi = 1. / (1 + np.exp(-p))
            else:
                exp_t = np.exp(p)
                phi = exp_t / (1. + exp_t)
            return (phi - 1) * b
        return partial_gradient

    def lipschitz_constant(self):
        return 0.25 * norm_rows(self.A) + self.alpha * self.A.shape[0]


class SquaredLoss:
    """Least squares loss function with L2 regularization"""

    def __init__(self, A, b, alpha='auto'):
        self.b = b
        self.A = A
        self.n_features = self.A.shape[1]
        if alpha == 'auto':
            self.alpha = 1. / A.shape[0]
        else:
            self.alpha = float(alpha)

    def __call__(self, x):
        # loss function to be optimized, it's the logistic loss
        z = self.A.dot(x) - self.b
        return 0.5 * (z * z).mean() + .5 * self.alpha * x.dot(x)

    def gradient(self, x):
        z = self.A.dot(x) - self.b
        grad_w = self.A.T.dot(z) / self.A.shape[0] + self.alpha * x
        return grad_w

    @staticmethod
    def partial_gradient_factory():
        @njit
        def partial_gradient(p, b):
            # compute p
            return - (b - p)
        return partial_gradient

    def lipschitz_constant(self):
        return norm_rows(self.A) + self.alpha * self.A.shape[0]


class L1Norm:
    """L1 norm, i.e., the sum of absolute values"""
    is_separable = True

    def __init__(self, alpha=1.):
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * np.sum(np.abs(x))

    def prox(self, x, step_size):
        return np.fmax(x - self.alpha * step_size, 0) \
            - np.fmax(- x - self.alpha * step_size, 0)

    def prox_factory(self):
        alpha = self.alpha
        @njit
        def prox_L1(x, step_size):
            return np.fmax(x - alpha * step_size, 0) \
                   - np.fmax(- x - alpha * step_size, 0)
        return prox_L1


class TotalVariation2D:
    """2-dimensional Total Variation pseudo-norm

    """

    def __init__(self, alpha, n_rows, n_cols, max_iter=100):
        self.alpha = alpha
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_iter = max_iter

    def __call__(self, x):
        img = x.reshape((self.n_rows, self.n_cols))
        tmp1 = np.abs(np.diff(img, axis=0))
        tmp2 = np.abs(np.diff(img, axis=1))
        return self.alpha * (tmp1.sum() + tmp2.sum())

    def prox(self, x, step_size):
        return prox_tv2d(
            step_size * self.alpha, x, self.n_rows, self.n_cols,
            max_iter=self.max_iter)


class DummyLoss:

    is_separable = True

    def __call__(self, x):
        return 0

    def prox(self, x, stepsize):
        return x

    def prox_factory(self):
        @njit
        def prox_dummy(x, step_size):
            return x
        return prox_dummy


def norm_rows(A):
    if sparse.issparse(A):
        return np.max(A.multiply(A).sum(1))
    else:
        return np.max((A * A).sum(1))
