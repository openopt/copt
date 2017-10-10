import numpy as np
from scipy import sparse, special, linalg
from scipy.sparse import linalg as splinalg
from numba import njit
from .tv_prox import prox_tv2d
import warnings


class LogisticLoss:
    """Logistic regression loss function with L2 regularization

    This loss function is very popular for binary classification tasks.
    Labels (b) are assumed to be 1 or -1.

    References
    ----------
    Loss function and gradients for full gradient methods are computed
    as detailed in to
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """

    def __init__(self, A, b, alpha=0, intercept=True):
        self.b = b
        self.A = splinalg.aslinearoperator(A)
        self.intercept = intercept
        if self.intercept == True:
            self.n_features = self.A.shape[1] + 1
        else:
            self.n_features = self.A.shape[1]
        self.alpha = float(alpha)

    def __call__(self, x):
        if self.intercept:
            x_, c = x[:-1], x[-1]
        else:
            x_, c = x, 0.
        z = self.A.matvec(x_) + c
        yz = self.b * z
        idx = yz > 0
        out = np.zeros_like(yz)
        out[idx] = np.log(1 + np.exp(-yz[idx]))
        out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
        out = out.mean() + .5 * self.alpha * x_.dot(x_)
        return out

    def gradient(self, x):
        if self.intercept:
            x_, c = x[:-1], x[-1]
        else:
            x_, c = x, 0.
        z = self.A.matvec(x_) + c
        z = special.expit(self.b * z)
        z0 = (z - 1) * self.b
        grad_w = self.A.rmatvec(z0) / self.A.shape[0] + self.alpha * x_
        grad_c = z0.mean()
        if self.intercept:
            return np.concatenate((grad_w, [grad_c]))
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

    def lipschitz_constant(self, kind='full'):
        if kind == 'samples':
            return 0.25 * norm_along_axis(self.A.A, 1) + self.alpha * self.A.shape[0]
        elif kind == 'full':
            from scipy.sparse.linalg import svds
            s = svds(self.A, k=1, return_singular_vectors=False)[0]
            return 0.25 * s * s / self.A.shape[0] + self.alpha
        elif kind == 'features':
            return 0.25 * norm_along_axis(self.A.A, 0) / self.A.shape[0] + self.alpha
        else:
            raise NotImplementedError


class SquaredLoss:
    """Least squares loss function with L2 regularization

    Parameters
    ----------
    A: ndarray or LinearOperator
        Design matrix. If None, it is taken as the identity
        matrix.

    b: ndarray

    alpha: float
        Amount of L2 regularization
    """

    def __init__(self, A, b, alpha=0, intercept=False):
        if intercept:
            raise NotImplementedError
        self.b = b
        if A is None:
            A = splinalg.LinearOperator(
                matvec=lambda x: x, rmatvec=lambda x: x,
                shape=(b.size, b.size)
            )
        self.A = splinalg.aslinearoperator(A)
        self.n_features = self.A.shape[1]
        self.alpha = float(alpha)

    def __call__(self, x):
        # loss function to be optimized, it's the logistic loss
        z = self.A.matvec(x) - self.b
        return .5 * (z * z).mean() + .5 * self.alpha * x.dot(x)

    def gradient(self, x):
        z = self.A.matvec(x) - self.b
        return self.A.rmatvec(z) / self.A.shape[0] + self.alpha * x

    @staticmethod
    def partial_gradient_factory():
        @njit
        def partial_gradient(p, b):
            # compute p
            return - (b - p)
        return partial_gradient

    def lipschitz_constant(self, kind='full'):
        if kind == 'samples':
            return norm_along_axis(self.A.A, 1) + self.alpha * self.A.shape[0]
        elif kind == 'full':
            s = splinalg.svds(self.A, k=1, return_singular_vectors=False)[0]
            return s * s / self.A.shape[0] + self.alpha
        elif kind == 'features':
            return norm_along_axis(self.A.A, 0) / self.A.shape[0] + self.alpha
        else:
            raise NotImplementedError


class L1Norm:
    """L1 norm, i.e., the sum of absolute values"""
    is_separable = True

    def __init__(self, alpha=1., intercept=True):
        self.alpha = alpha
        self.intercept = intercept

    def __call__(self, x):
        x_ = x
        if self.intercept:
            x_ = x[:-1]
        return self.alpha * np.sum(np.abs(x_))

    def prox(self, x, step_size):
        x_ = x
        if self.intercept:
            x_ = x[:-1]
        out = np.fmax(x_ - self.alpha * step_size, 0) \
            - np.fmax(- x_ - self.alpha * step_size, 0)
        if self.intercept:
            return np.concatenate((out, (x_[-1],)))
        return out

    def prox_factory(self):
        @njit
        def prox_L1(x, step_size):
            return np.fmax(x - self.alpha * step_size, 0) \
                   - np.fmax(- x - self.alpha * step_size, 0)
        return prox_L1


class L1Ball:
    """Projection onto the L1 ball"""
    is_separable = True

    def __init__(self, alpha=1., intercept=True):
        self.alpha = alpha
        self.intercept = intercept

    def __call__(self, x):
        x_ = x
        if self.intercept:
            x_ = x[:-1]
        if np.abs(x_).sum() <= self.alpha:
            return 0
        else:
            return np.inf

    def prox(self, x, step_size):
        x_ = x
        if self.intercept:
            x_ = x[:-1]

        out = euclidean_proj_l1ball(x_, self.alpha)
        if self.intercept:
            return np.concatenate((out, (x_[-1],)))
        return out

    def prox_factory(self):
        raise NotImplementedError



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




class TraceNorm:
    """Trace (aka nuclear) norm, sum of singular values"""
    is_separable = False

    def __init__(self, shape, alpha=1.):
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

    def __init__(self, shape, alpha=1.):
        assert len(shape) == 2
        self.shape = shape
        self.alpha = alpha

    def __call__(self, x):
        X = x.reshape(self.shape)
        if linalg.svdvals(X).sum() <= self.alpha:
            return 0
        else:
            return np.inf

    def prox(self, x, step_size):
        X = x.reshape(self.shape)
        U, s, Vt = linalg.svd(X, full_matrices=False)
        s_threshold = euclidean_proj_l1ball(s, self.alpha)
        return (U * s_threshold).dot(Vt).ravel()
        # try:
        #     U, s, Vt = linalg.svd(X, full_matrices=False)
        #     s_threshold = euclidean_proj_l1ball(s, self.alpha)
        #     return (U * s_threshold).dot(Vt).ravel()
        # except:
        #     warnings.warn('SVD failed')

    def prox_factory(self):
        raise NotImplementedError


class TotalVariation2D:
    """2-dimensional Total Variation pseudo-norm"""

    def __init__(self, alpha, n_rows, n_cols, max_iter=100, tol=1e-6):
        self.alpha = alpha
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_iter = max_iter
        self.tol = tol

    def __call__(self, x):
        img = x.reshape((self.n_rows, self.n_cols))
        tmp1 = np.abs(np.diff(img, axis=0))
        tmp2 = np.abs(np.diff(img, axis=1))
        return self.alpha * (tmp1.sum() + tmp2.sum())

    def prox(self, x, step_size):
        return prox_tv2d(
            step_size * self.alpha, x, self.n_rows, self.n_cols,
            max_iter=self.max_iter, tol=self.tol)


class ZeroLoss:

    is_separable = True

    def __call__(self, x):
        return 0

    def prox(self, x, step_size):
        return x

    def prox_factory(self):
        @njit
        def prox_dummy(x, step_size):
            return x
        return prox_dummy


def norm_along_axis(A, axis=1):
    if sparse.issparse(A):
        return np.max(A.multiply(A).sum(axis))
    else:
        return np.max((A * A).sum(axis))
