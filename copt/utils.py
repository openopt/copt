import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as splinalg
from numba import njit
from datetime import datetime
from sklearn.utils.extmath import safe_sparse_dot

from . import tv_prox

class Trace:
    def __init__(self, freq=1):
        self.trace_x = []
        self.trace_time = []
        self.start = datetime.now()
        self._counter = 0
        self.freq = int(freq)

    def __call__(self, x):
        if self._counter % self.freq == 0:
            self.trace_x.append(x.copy())
            delta = (datetime.now() - self.start).total_seconds()
            self.trace_time.append(delta)
        self._counter += 1


def get_lipschitz(A, loss, alpha=0):
    """Estimate Lipschitz constant for different loss functions

    A : array-like

    loss : {'logloss', 'square', 'huber'}
    """

    if hasattr(loss, 'name'):
        loss = loss.name

    if loss == 'logloss':
        s = splinalg.svds(A, k=1, return_singular_vectors=False,
                          maxiter=100)[0]
        return 0.25 * (s * s) / A.shape[0] + alpha
    elif loss in ('huber', 'square'):
        s = splinalg.svds(A, k=1, return_singular_vectors=False,
                          maxiter=100)[0]
        return (s * s) / A.shape[0] + alpha
    raise NotImplementedError


def get_max_lipschitz(A, loss, alpha=0):
    """Estimate the max Lipschitz constant (as appears in
    many stochastic methods).

    A : array-like

    loss : {'logloss', 'square', 'huber'}
    """
    from sklearn.utils.extmath import row_norms
    max_squared_sum = row_norms(A, squared=True).max()

    if loss == 'logloss':
        return 0.25 * max_squared_sum + alpha
    elif loss in ('huber', 'square'):
        raise NotImplementedError
    raise NotImplementedError



class LogLoss:
    """A class evaluation and derivatives of the logistic loss. The logistic loss function is defined as

    .. math::
        -\\frac{1}{n}\\sum_{i=1}^n b_i \\log(\sigma(a_i^T x)) + (1 - b_i) \\log(1 - \sigma(a_i^T x))

    where :math:`\sigma` is the sigmoid function :math:`\sigma(t) = 1/(1 + e^{-t})`.
    
    When the input vector b comes from class labels, it is expected to have the values 0 or 1.

    for a numerically stable computation of the logistic loss, we use the identities

    .. math::
        \log(\sigma(t)) = \\begin{cases} -\log(1 + e^{-t}) &\\text{ if $t \geq 0$}\\\\
        t - \log(1 + e^t) &\\text{ otherwise}\end{cases}

        \log(1 - \sigma(t)) = \\begin{cases} -t -\log(1 + e^{-t}) &\\text{ if $t \geq 0$}\\\\
        - \log(1 + e^t) &\\text{ otherwise}\end{cases}


    """
    def __init__(self, A, b, alpha=0.):
        if A is None:
            A = sparse.eye(b.size, b.size, format='csr')
        self.A = A
        if np.max(b) > 1 or np.min(b) < 0:
            raise ValueError('b can only contain values between 0 and 1 ')
        if not A.shape[0] == b.size:
            raise ValueError('Dimensions of A and b do not coincide')
        self.b = b
        self.alpha = alpha
        self.intercept = False
        self.name = 'logloss'

    def __call__(self, x):
        return self.f_grad(x, return_gradient=False)

    def f_grad(self, x, return_gradient=True):
        if self.intercept:
            x_, c = x[:-1], x[-1]
        else:
            x_, c = x, 0.
        z = safe_sparse_dot(self.A, x_, dense_output=True).ravel() + c
        idx = z > 0
        loss_vec = np.zeros_like(z)
        loss_vec[idx] = np.log(1 + np.exp(-z[idx])) + (1 - self.b[idx]) * z[idx]
        loss_vec[~idx] = np.log(1 + np.exp(z[~idx])) - self.b[~idx] * z[~idx]
        loss = loss_vec.mean() + .5 * self.alpha * safe_sparse_dot(x_.T, x_, dense_output=True).ravel()[0]

        if not return_gradient:
            return loss
        z0 = np.zeros_like(z)
        tmp = np.exp(-z[idx])
        z0[idx] = - tmp / (1 + tmp) + 1 - self.b[idx]
        tmp = np.exp(z[~idx])
        z0[~idx] = tmp / (1 + tmp) - self.b[~idx]
        grad = self.A.T.dot(z0) / self.A.shape[0] + self.alpha * x_.T
        grad = np.asarray(grad).ravel()
        grad_c = z0.mean()
        if self.intercept:
            return np.concatenate((grad, [grad_c]))

        return loss, grad


class SquareLoss:
    def __init__(self, A, b, alpha=0):
        if A is None:
            A = sparse.eye(b.size, b.size, format='csr')
        self.b = b
        self.alpha = alpha
        self.A = A
        self.name = 'square'

    def __call__(self, x):
        z = safe_sparse_dot(self.A, x, dense_output=True).ravel() - self.b
        return 0.5 * (z * z).mean() + .5 * self.alpha * safe_sparse_dot(x.T, x, dense_output=True).ravel()[0]

    def f_grad(self, x, return_gradient=True):
        z = safe_sparse_dot(self.A, x, dense_output=True).ravel() - self.b
        loss = 0.5 * (z * z).mean() + .5 * self.alpha * safe_sparse_dot(x.T, x, dense_output=True).ravel()[0]
        if not return_gradient:
            return loss
        grad = self.A.T.dot(z) / self.A.shape[0] + self.alpha * x.T
        return loss, np.asarray(grad).ravel()


class HuberLoss:
    """Huber loss"""
    def __init__(self, A, b, alpha=0, delta=1):
        self.delta = delta
        self.A = A
        self.b = b
        self.alpha = alpha
        self.name = 'huber'

    def __call__(self, x):
        return self.f_grad(x, return_gradient=False)

    def f_grad(self, x, return_gradient=True):
        z = safe_sparse_dot(self.A, x, dense_output=True).ravel() - self.b
        idx = np.abs(z) < self.delta
        loss = 0.5 * np.sum(z[idx] * z[idx])
        loss += np.sum(self.delta * (np.abs(z[~idx]) - 0.5 * self.delta))
        loss = loss / z.size + .5 * self.alpha * safe_sparse_dot(x.T, x, dense_output=True).ravel()[0]
        if not return_gradient:
            return loss
        grad = self.A[idx].T.dot(z[idx]) / self.A.shape[0] + self.alpha * x.T
        grad = np.asarray(grad)
        grad += self.A[~idx].T.dot(self.delta * np.sign(z[~idx]))/ self.A.shape[0]
        return loss, grad


def ilogloss():

    @njit
    def partial_f_grad(p, b):
        p *= b
        if p > 0:
            tmp = 1 + np.exp(-p)
            return np.log(tmp), 1. / tmp
        else:
            exp_t = np.exp(p)
            phi = exp_t / (1. + exp_t)
            return -p + np.log(exp_t), (phi - 1) * b


class L1Norm:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * np.abs(x).sum()

    def prox(self, x, step_size):
        return np.fmax(x - self.alpha * step_size, 0) \
                   - np.fmax(- x - self.alpha * step_size, 0)


class NuclearNorm:
    def __init__(self, alpha, n_rows, n_cols):
        self.alpha = alpha
        self.shape = (n_rows, n_cols)

    def __call__(self, x):
        X = x.reshape(self.shape)
        U, s, Vt = linalg.svd(X, full_matrices=False)
        return self.alpha * np.sum(np.abs(s))

    def prox(self, x, step_size):
        X = x.reshape(self.shape)
        U, s, Vt = linalg.svd(X, full_matrices=False)
        s_threshold = L1Norm(self.alpha).prox(s, step_size)
        return (U * s_threshold).dot(Vt).ravel()


class L1Ball:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        if np.abs(x).sum() <= self.alpha:
            return 0
        else:
            return np.infty

    def prox(self, x, step_size):
        return euclidean_proj_l1ball(x, self.alpha)

    def lmo(self, u):
        """Solve the linear problem
        min_{||s||_1 <= alpha} <u, s>
        """
        idx = np.argmax(np.abs(u))
        mag = self.alpha * np.sign(u[idx])
        s_data = np.array([mag])
        s_indices = np.array([idx], dtype=np.int32)
        s_indptr = np.array([0, 1], dtype=np.int32)
        return sparse.csr_matrix((s_data, s_indices, s_indptr), shape=(1, u.size)).T


class GroupL1:
    """
    Group Lasso penalty
    
    
    XXX TODO define blocks
    """
    def __init__(self, alpha, blocks):
        self.alpha = alpha
        self.n_features = len(blocks)
        self.groups = [np.where(blocks == b)[0] for b in np.unique(blocks)]

    def __call__(self, x):
        return self.alpha * np.sum(
            [np.linalg.norm(x[g]) for g in self.groups])

    def prox(self, x, step_size):
        if self.n_features != x.size:
            raise ValueError('Dimensions of blocks and x do not match')
        out = x.copy()
        for g in self.groups:

            norm = np.linalg.norm(x[g])
            if norm > self.alpha * step_size:
                out[g] -= step_size * self.alpha * out[g] / norm
            else:
                out[g] = 0
            # if norm > 0:
            #     scaling = np.fmax(1 - self.alpha * step_size / norm, 0)
            #     out[g] *= scaling
        return out


class SimplexConstraint:
    def __init__(self, s=1):
        self.s = s

    def prox(self, x, step_size):
        return euclidean_proj_simplex(x, self.s)
#
# def grad_squareloss(A, b, alpha=0.):
#     """
#
#     Parameters
#     ----------
#     A
#     b
#     alpha
#     intercept
#
#     Returns
#     -------
#     logloss : callable
#     """
#     A = splinalg.aslinearoperator(A)
#
#     def _squareloss_func(x):
#         z = A.matvec(x) - b
#         loss = 0.5 * (z * z).mean() + .5 * alpha * x.dot(x)
#         return loss
#
#
#     def _squareloss_grad(x, return_gradient=True):
#         z = A.matvec(x) - b
#         loss = 0.5 * (z * z).mean() + .5 * alpha * x.dot(x)
#         if not return_gradient:
#             return loss
#         grad = A.rmatvec(z) / A.shape[0] + alpha * x
#         return loss, grad
#     return _squareloss_grad


def euclidean_proj_simplex(v, s=1.):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: float, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s=1,):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: float, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w


class TraceNorm:
    """Trace (aka nuclear) norm, sum of singular values"""
    is_separable = False

    def __init__(self, alpha, shape):
        assert len(shape) == 2
        self.shape = shape
        self.alpha = alpha

    def __call__(self, x):
        X = x.reshape(self.shape)
        return self.alpha * linalg.svdvals(X).sum()

    def prox(self, x, step_size):
        X = x.reshape(self.shape)
        U, s, Vt = linalg.svd(X, full_matrices=False)
        s_threshold = np.fmax(s - self.alpha * step_size, 0) \
            - np.fmax(- s - self.alpha * step_size, 0)
        return (U * s_threshold).dot(Vt).ravel()

    def prox_factory(self):
        raise NotImplementedError


class TraceBall:
    """Projection onto the trace (aka nuclear) norm, sum of singular values"""
    is_separable = False

    def __init__(self, alpha, shape):
        assert len(shape) == 2
        self.shape = shape
        self.alpha = alpha

    def __call__(self, x):
        X = x.reshape(self.shape)
        if linalg.svdvals(X).sum() <= self.alpha + np.finfo(np.float32).eps:
            return 0
        else:
            return np.inf

    def prox(self, x, step_size):
        X = x.reshape(self.shape)
        U, s, Vt = linalg.svd(X, full_matrices=False)
        s_threshold = euclidean_proj_l1ball(s, self.alpha)
        return (U * s_threshold).dot(Vt).ravel()

    def prox_factory(self):
        raise NotImplementedError


class TotalVariation2D:
    """2-dimensional Total Variation pseudo-norm"""

    def __init__(self, alpha, shape, max_iter=100, tol=1e-6):
        self.alpha = alpha
        self.n_rows = shape[0]
        self.n_cols = shape[1]
        self.max_iter = max_iter
        self.tol = tol

    def __call__(self, x):
        img = x.reshape((self.n_rows, self.n_cols))
        tmp1 = np.abs(np.diff(img, axis=0))
        tmp2 = np.abs(np.diff(img, axis=1))
        return self.alpha * (tmp1.sum() + tmp2.sum())

    def prox(self, x, step_size):
        return tv_prox.prox_tv2d(
            step_size * self.alpha, x, self.n_rows, self.n_cols,
            max_iter=self.max_iter, tol=self.tol)
