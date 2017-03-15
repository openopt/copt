import numpy as np
from scipy import sparse
from numba import njit
from copt.utils import norm_rows


class SmoothLoss:

    def gradient(self, x):
        raise NotImplementedError


class LogisticLoss(SmoothLoss):

    def __init__(self, A, b, alpha=1.0):
        # A = sparse.csr_matrix(A)
        self.b = b
        self.alpha = alpha
        self.A = sparse.csr_matrix(A)

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
        return 0.25 * norm_rows(self.A) + self.alpha


class NormL1:

    def __init__(self, alpha=1.):
        self.is_separable = True
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * np.sum(np.abs(x))

    def prox(self, x, step_size):
        return np.fmax(x - self.alpha * step_size, 0) \
            - np.fmax(- x - self.alpha * step_size, 0)

    def block_prox_factory(self):
        alpha = self.alpha
        @njit
        def prox_L1(x, step_size):
            return np.fmax(x - alpha * step_size, 0) \
                   - np.fmax(- x - alpha * step_size, 0)
        return prox_L1

class DummyProx:

    def __call__(self, x):
        return 0

    def prox(self, x, stepsize):
        return x

    def block_prox_factory(self):
        @njit
        def prox_dummy(x, step_size):
            return x
        return prox_dummy
