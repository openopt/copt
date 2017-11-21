import numpy as np
from scipy import sparse, special, linalg
from scipy.sparse import linalg as splinalg
from numba import njit
from sklearn.utils.extmath import row_norms
from .tv_prox import prox_tv2d
import warnings
from datetime import datetime


class Trace:
    def __init__(self):
        self.trace_x = []
        self.trace_time = []
        self.start = datetime.now()

    def __call__(self, kw):
        self.trace_x.append(kw['x'].copy())
        delta = (datetime.now() - self.start).total_seconds()
        self.trace_time.append(delta)


def get_step_size(A, loss, alpha=0, method='PGD'):
    if loss == 'log':
        s = splinalg.svds(A, k=1, return_singular_vectors=False,
                          tol=1e-2, maxiter=20)[0]
        L = 0.25 * (s * s) / A.shape[0] + alpha
        return 1/L
    elif loss == 'square':
        s = splinalg.svds(A, k=1, return_singular_vectors=False,
                          tol=1e-2, maxiter=20)[0]
        L = (s * s) / A.shape[0] + alpha
        return 1/L
    raise NotImplementedError

def lipschitz_constant(A, loss, alpha=0):
    """Computes the Lipschitz constant"""
    if loss == 'log':
        pass
    elif loss == 'square':

        return


def logloss(A, b, alpha=0., intercept=False):
    """

    Parameters
    ----------
    A
    b
    alpha
    intercept

    Returns
    -------
    logloss : callable
    """
    A = splinalg.aslinearoperator(A)

    def _logloss_grad(x, return_gradient=True):
        if intercept:
            x_, c = x[:-1], x[-1]
        else:
            x_, c = x, 0.
        z = A.matvec(x_) + c
        yz = b * z
        idx = yz > 0
        loss = np.zeros_like(yz)
        loss[idx] = np.log(1 + np.exp(-yz[idx]))
        loss[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
        loss = loss.mean() + .5 * alpha * x_.dot(x_)

        if not return_gradient:
            return loss
        z = special.expit(b * z)
        z0 = (z - 1) * b
        grad = A.rmatvec(z0) / A.shape[0] + alpha * x_
        grad_c = z0.mean()
        if intercept:
            return np.concatenate((grad, [grad_c]))

        return loss, grad
    return _logloss_grad


def ilogloss():

    @njit
    def partial_func_grad(p, b):
        p *= b
        if p > 0:
            tmp = 1 + np.exp(-p)
            return np.log(tmp), 1. / tmp
        else:
            exp_t = np.exp(p)
            phi = exp_t / (1. + exp_t)
            return -p + np.log(exp_t), (phi - 1) * b


def prox_L1(alpha):
    def _prox_L1(x, step_size):
        return np.fmax(x - alpha * step_size, 0) \
               - np.fmax(- x - alpha * step_size, 0)
    return _prox_L1




def squareloss(A, b, alpha=0.):
    """

    Parameters
    ----------
    A
    b
    alpha
    intercept

    Returns
    -------
    logloss : callable
    """
    A = splinalg.aslinearoperator(A)

    def _squareloss_grad(x, return_gradient=True):
        z = A.matvec(x) - b
        loss = 0.5 * (z * z).mean() + .5 * alpha * x.dot(x)
        if not return_gradient:
            return loss
        grad = A.rmatvec(z) / A.shape[0] + alpha * x
        return loss, grad
    return _squareloss_grad



# class LogisticLoss(_GeneralizedLinearModel):
#     """Logistic regression loss function with L2 regularization
#
#     This loss function is very popular for binary classification tasks.
#     Labels (b) are assumed to be 1 or -1.
#
#     References
#     ----------
#     Loss function and gradients for full gradient methods are computed
#     as detailed in to
#     http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
#     """
#
#     def __call__(self, x):
#         if self.intercept:
#             x_, c = x[:-1], x[-1]
#         else:
#             x_, c = x, 0.
#         z = self.A.matvec(x_) + c
#         yz = self.b * z
#         idx = yz > 0
#         out = np.zeros_like(yz)
#         out[idx] = np.log(1 + np.exp(-yz[idx]))
#         out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
#         out = out.mean() + .5 * self.alpha * x_.dot(x_)
#         return out
#
#     def gradient(self, x):
#         if self.intercept:
#             x_, c = x[:-1], x[-1]
#         else:
#             x_, c = x, 0.
#         z = self.A.matvec(x_) + c
#         z = special.expit(self.b * z)
#         z0 = (z - 1) * self.b
#         grad_w = self.A.rmatvec(z0) / self.A.shape[0] + self.alpha * x_
#         grad_c = z0.mean()
#         if self.intercept:
#             return np.concatenate((grad_w, [grad_c]))
#         return grad_w
#
#     @staticmethod
#     def partial_function_factory():
#         @njit
#         def partial_function(p, b):
#             # compute p
#             p *= b
#             if p > 0:
#                 return np.log(1 + np.exp(-p))
#             else:
#                 return -p + np.log(1 + np.exp(p))
#
#     @staticmethod
#     def partial_gradient_factory():
#         @njit
#         def partial_gradient(p, b):
#             # compute p
#             p *= b
#             if p > 0:
#                 phi = 1. / (1 + np.exp(-p))
#             else:
#                 exp_t = np.exp(p)
#                 phi = exp_t / (1. + exp_t)
#             return (phi - 1) * b
#         return partial_gradient
#
#     def lipschitz_constant(self, kind='full'):
#         if kind == 'samples':
#             return 0.25 * row_norms(self.A.A, True).max() + self.alpha * self.A.shape[0]
#         elif kind == 'full':
#             from scipy.sparse.linalg import svds
#             s = svds(self.A, k=1, return_singular_vectors=False)[0]
#             return 0.25 * s * s / self.A.shape[0] + self.alpha
#         elif kind == 'features':
#             return 0.25 * row_norms(self.A.A.T, True).max() / self.A.shape[0] + self.alpha
#         else:
#             raise NotImplementedError
#
#
# class SquaredLoss(_GeneralizedLinearModel):
#     """Least squares loss function with L2 regularization
#
#     Parameters
#     ----------
#     A: ndarray or LinearOperator
#         Design matrix. If None, it is taken as the identity
#         matrix.
#
#     b: ndarray
#
#     alpha: float
#         Amount of L2 regularization
#     """
#
#     def __call__(self, x):
#         if self.intercept:
#             x_, c = x[:-1], x[-1]
#             z = self.A.matvec(x_) + c - self.b
#         else:
#             z = self.A.matvec(x) - self.b
#
#         return .5 * (z * z).mean() + .5 * self.alpha * x.dot(x)
#
#     def gradient(self, x):
#         if self.intercept:
#             x_, c = x[:-1], x[-1]
#         else:
#             x_, c = x, 0.
#         z = self.A.matvec(x) - self.b
#         return self.A.rmatvec(z) / self.A.shape[0] + self.alpha * x
#
#     @staticmethod
#     def partial_gradient_factory():
#         @njit
#         def partial_gradient(p, b):
#             # compute p
#             return - (b - p)
#         return partial_gradient
#
#     def lipschitz_constant(self, kind='full'):
#         if kind == 'samples':
#             return row_norms(self.A.A, True).max() + self.alpha * self.A.shape[0]
#         elif kind == 'full':
#             while True:
#                 try:
#                     s = splinalg.svds(
#                         self.A, k=1,
#                         return_singular_vectors=False,
#                         tol=1e-6, maxiter=100)[0]
#                     break
#                 except splinalg.ArpackError:
#                     # should we do something like increasing
#                     # ncv as the error message suggests?
#                     # doing nothing seems to fix the issue ...
#                     pass
#             return s * s / self.A.shape[0] + self.alpha
#         elif kind == 'features':
#             return row_norms(self.A.A, True).max() / self.A.shape[0] + self.alpha
#         else:
#             raise NotImplementedError
#
#
# class L1Norm:
#     """L1 norm, i.e., the sum of absolute values"""
#     is_separable = True
#
#     def __init__(self, alpha=1.):
#         self.alpha = alpha
#
#     def __call__(self, x):
#         return self.alpha * np.sum(np.abs(x))
#
#     def prox(self, x, step_size):
#         out = np.fmax(x - self.alpha * step_size, 0) \
#             - np.fmax(- x - self.alpha * step_size, 0)
#         return out
#
#     def prox_factory(self):
#         @njit
#         def prox_L1(x, step_size):
#             return np.fmax(x - self.alpha * step_size, 0) \
#                    - np.fmax(- x - self.alpha * step_size, 0)
#         return prox_L1
#
#
# class L1Ball:
#     """Projection onto the L1 ball"""
#     is_separable = True
#
#     def __init__(self, alpha=1.):
#         self.alpha = alpha
#
#     def __call__(self, x):
#         if np.abs(x).sum() <= self.alpha:
#             return 0
#         else:
#             return np.inf
#
#     def prox(self, x, step_size):
#         out = euclidean_proj_l1ball(x, self.alpha)
#         return out
#
#     def prox_factory(self):
#         raise NotImplementedError
#


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


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
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
#
#
# class TraceNorm:
#     """Trace (aka nuclear) norm, sum of singular values"""
#     is_separable = False
#
#     def __init__(self, shape, alpha=1.):
#         assert len(shape) == 2
#         self.shape = shape
#         self.alpha = alpha
#
#     def __call__(self, x):
#         X = x.reshape(self.shape)
#         return self.alpha * linalg.svdvals(X).sum()
#
#     def prox(self, x, step_size):
#         X = x.reshape(self.shape)
#         U, s, Vt = linalg.svd(X, full_matrices=False)
#         s_threshold = np.fmax(s - self.alpha * step_size, 0) \
#             - np.fmax(- s - self.alpha * step_size, 0)
#         return (U * s_threshold).dot(Vt).ravel()
#
#     def prox_factory(self):
#         raise NotImplementedError
#
#
# class TraceBall:
#     """Projection onto the trace (aka nuclear) norm, sum of singular values"""
#     is_separable = False
#
#     def __init__(self, shape, alpha=1.):
#         assert len(shape) == 2
#         self.shape = shape
#         self.alpha = alpha
#
#     def __call__(self, x):
#         X = x.reshape(self.shape)
#         if linalg.svdvals(X).sum() <= self.alpha:
#             return 0
#         else:
#             return np.inf
#
#     def prox(self, x, step_size):
#         try:
#             X = x.reshape(self.shape)
#             U, s, Vt = linalg.svd(X, full_matrices=False)
#             s_threshold = euclidean_proj_l1ball(s, self.alpha)
#             return (U * s_threshold).dot(Vt).ravel()
#         except linalg.LinAlgError:
#             # SVD did not converge
#             warnings.warn('SVD failed')
#             return x
#
#     def prox_factory(self):
#         raise NotImplementedError
#
#
# class TotalVariation2D:
#     """2-dimensional Total Variation pseudo-norm"""
#
#     def __init__(self, alpha, n_rows, n_cols, max_iter=100, tol=1e-6):
#         self.alpha = alpha
#         self.n_rows = n_rows
#         self.n_cols = n_cols
#         self.max_iter = max_iter
#         self.tol = tol
#
#     def __call__(self, x):
#         img = x.reshape((self.n_rows, self.n_cols))
#         tmp1 = np.abs(np.diff(img, axis=0))
#         tmp2 = np.abs(np.diff(img, axis=1))
#         return self.alpha * (tmp1.sum() + tmp2.sum())
#
#     def prox(self, x, step_size):
#         return prox_tv2d(
#             step_size * self.alpha, x, self.n_rows, self.n_cols,
#             max_iter=self.max_iter, tol=self.tol)
#
#
