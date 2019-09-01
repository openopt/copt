import numpy as np
import numpy.ma as ma
from scipy import sparse
from scipy import linalg
from scipy import special
from scipy.sparse import linalg as splinalg
from datetime import datetime
from sklearn.utils.extmath import safe_sparse_dot


try:
    from numba import njit
except ImportError:
    from functools import wraps

    def njit(*args, **kw):
        if len(args) == 1 and len(kw) == 0 and hasattr(args[0], "__call__"):
            func = args[0]

            @wraps(func)
            def inner_function(*args, **kwargs):
                return func(*args, **kwargs)

            return inner_function
        else:

            def inner_function(function):
                @wraps(function)
                def wrapper(*args, **kwargs):
                    return function(*args, **kwargs)

                return wrapper

            return inner_function


def safe_sparse_add(a, b):
    if sparse.issparse(a) and sparse.issparse(b):
        # both are sparse, keep the result sparse
        return a + b
    else:
        # on of them is non-sparse, convert
        # everything to dense.
        if sparse.issparse(a):
            a = a.toarray()
            if a.ndim == 2 and b.ndim == 1:
                b.ravel()
        elif sparse.issparse(b):
            b = b.toarray()
            if b.ndim == 2 and a.ndim == 1:
                b = b.ravel()
        return a + b


def parse_step_size(step_size):
    if hasattr(step_size, "__len__") and len(step_size) == 2:
        return step_size[0], step_size[1]
    elif isinstance(step_size, float):
        return step_size, "fixed"
    elif hasattr(step_size, "__call__") or step_size == "adaptive":
        # without other information start with a step-size of one
        return 1, step_size
    else:
        raise ValueError("Could not understand value step_size=%s" % step_size)


class Trace:
    def __init__(self, f=None, freq=1):
        self.trace_x = []
        self.trace_time = []
        self.trace_fx = []
        self.trace_step_size = []
        self.start = datetime.now()
        self._counter = 0
        self.freq = int(freq)
        self.f = f

    def __call__(self, dl):
        if self._counter % self.freq == 0:
            if self.f is not None:
                self.trace_fx.append(self.f(dl["x"]))
            else:
                self.trace_x.append(dl["x"].copy())
            delta = (datetime.now() - self.start).total_seconds()
            self.trace_time.append(delta)
            self.trace_step_size.append(dl["step_size"])
        self._counter += 1


def init_lipschitz(f_grad, x0):
    L0 = 1e-3
    f0, grad0 = f_grad(x0)
    if sparse.issparse(grad0) and not sparse.issparse(x0):
        x0 = sparse.csc_matrix(x0).T
    elif sparse.issparse(x0) and not sparse.issparse(grad0):
        grad0 = sparse.csc_matrix(grad0).T
    x_tilde = x0 - (1.0 / L0) * grad0
    f_tilde = f_grad(x_tilde)[0]
    for _ in range(100):
        if f_tilde <= f0:
            break
        L0 *= 10
        x_tilde = x0 - (1.0 / L0) * grad0
        f_tilde = f_grad(x_tilde)[0]
    return L0


def get_max_lipschitz(A, loss, alpha=0):
    """
  XXX DEPRECATED

  Estimate the max Lipschitz constant (as appears in
  many stochastic methods).

  A : array-like

  loss : {"logloss", "square", "huber"}
  """
    from sklearn.utils.extmath import row_norms

    max_squared_sum = row_norms(A, squared=True).max()

    if loss == "logloss":
        return 0.25 * max_squared_sum + alpha
    elif loss in ("huber", "square"):
        raise NotImplementedError
    raise NotImplementedError


class LogLoss:
    r"""Logistic loss function.

  The logistic loss function is defined as

  .. math::
      -\frac{1}{n}\sum_{i=1}^n b_i \log(\sigma(\bs{a}_i^T \bs{x}))
         + (1 - b_i) \log(1 - \sigma(\bs{a}_i^T \bs{x}))

  where :math:`\sigma` is the sigmoid function
  :math:`\sigma(t) = 1/(1 + e^{-t})`.

  The input vector b verifies :math:`0 \leq b_i \leq 1`. When it comes from
  class labels, it should have the values 0 or 1.

  References:
    http://fa.bianp.net/drafts/derivatives_logistic.html
  """

    def __init__(self, A, b, alpha=0.0):
        if A is None:
            A = sparse.eye(b.size, b.size, format="csr")
        self.A = A
        if np.max(b) > 1 or np.min(b) < 0:
            raise ValueError("b can only contain values between 0 and 1 ")
        if not A.shape[0] == b.size:
            raise ValueError("Dimensions of A and b do not coincide")
        self.b = b
        self.alpha = alpha
        self.intercept = False

    def __call__(self, x):
        return self.f_grad(x, return_gradient=False)

    def _sigma(self, z, idx):
        z0 = np.zeros_like(z)
        tmp = np.exp(-z[idx])
        z0[idx] = 1 / (1 + tmp)
        tmp = np.exp(z[~idx])
        z0[~idx] = tmp / (1 + tmp)
        return z0

    def logsig(self, x):
        """Compute log(1 / (1 + exp(-t))) component-wise."""
        out = np.zeros_like(x)
        idx0 = x < -33
        out[idx0] = x[idx0]
        idx1 = (x >= -33) & (x < -18)
        out[idx1] = x[idx1] - np.exp(x[idx1])
        idx2 = (x >= -18) & (x < 37)
        out[idx2] = -np.log1p(np.exp(-x[idx2]))
        idx3 = x >= 37
        out[idx3] = -np.exp(-x[idx3])
        return out

    def expit_b(self, x, b):
        """Compute sigmoid(x) - b."""
        idx = x < 0
        out = np.zeros_like(x)
        exp_x = np.exp(x[idx])
        b_idx = b[idx]
        out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
        exp_nx = np.exp(-x[~idx])
        b_nidx = b[~idx]
        out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
        return out

    def f_grad(self, x, return_gradient=True):
        if self.intercept:
            x_, c = x[:-1], x[-1]
        else:
            x_, c = x, 0.0
        z = safe_sparse_dot(self.A, x_, dense_output=True).ravel() + c
        loss = np.mean((1 - self.b) * z - self.logsig(z))
        penalty = safe_sparse_dot(x_.T, x_, dense_output=True).ravel()[0]
        loss += 0.5 * self.alpha * penalty

        if not return_gradient:
            return loss

        z0_b = self.expit_b(z, self.b)

        grad = safe_sparse_add(self.A.T.dot(z0_b) / self.A.shape[0], self.alpha * x_)
        grad = np.asarray(grad).ravel()
        grad_c = z0_b.mean()
        if self.intercept:
            return np.concatenate((grad, [grad_c]))

        return loss, grad

    def Hessian(self, x):
        """Return a callable that performs dot products with the Hessian."""

        n_samples, n_features = self.A.shape
        if self.intercept:
            x_, c = x[:-1], x[-1]
        else:
            x_, c = x, 0.0

        z = special.expit(safe_sparse_dot(self.A, x_, dense_output=True).ravel() + c)

        # The mat-vec product of the Hessian
        d = z * (1 - z)
        if sparse.issparse(self.A):
            dX = safe_sparse_dot(
                sparse.dia_matrix((d, 0), shape=(n_samples, n_samples)), self.A
            )
        else:
            # Precompute as much as possible
            dX = d[:, np.newaxis] * self.A

        if self.intercept:
            # Calculate the double derivative with respect to intercept
            # In the case of sparse matrices this returns a matrix object.
            dd_intercept = np.squeeze(np.array(dX.sum(axis=0)))

        def _Hs(s):
            ret = np.empty_like(s)
            ret[:n_features] = self.A.T.dot(dX.dot(s[:n_features]))
            ret[:n_features] += self.alpha * s[:n_features]

            # For the fit intercept case.
            if self.intercept:
                ret[:n_features] += s[-1] * dd_intercept
                ret[-1] = dd_intercept.dot(s[:n_features])
                ret[-1] += d.sum() * s[-1]
            return ret / n_samples

        return _Hs

    @property
    def partial_deriv(self):
        @njit
        def log_deriv(p, y):
            # derivative of logistic loss
            # same as in lightning (with minus sign)
            if p > 0:
                tmp = np.exp(-p)
                phi = -tmp / (1.0 + tmp) + 1 - y
            else:
                tmp = np.exp(p)
                phi = tmp / (1.0 + tmp) - y
            return phi

        return log_deriv

    @property
    def lipschitz(self):
        s = splinalg.svds(self.A, k=1, return_singular_vectors=False)[0]
        return 0.25 * (s * s) / self.A.shape[0] + self.alpha

    @property
    def max_lipschitz(self):
        from sklearn.utils.extmath import row_norms

        max_squared_sum = row_norms(self.A, squared=True).max()

        return 0.25 * max_squared_sum + self.alpha


class SquareLoss:
    r"""Squared loss.

  The Squared loss is defined as

  .. math::
      \frac{1}{n}\|A x - b\|^2~,

  where :math:`\|\cdot\|` is the euclidean norm.
  """

    def __init__(self, A, b, alpha=0):
        if A is None:
            A = sparse.eye(b.size, b.size, format="csr")
        self.b = b
        self.alpha = alpha
        self.A = A
        self.name = "square"

    def __call__(self, x):
        z = safe_sparse_dot(self.A, x, dense_output=True).ravel() - self.b
        pen = self.alpha * safe_sparse_dot(x.T, x, dense_output=True).ravel()[0]
        return 0.5 * (z * z).mean() + 0.5 * pen

    def f_grad(self, x, return_gradient=True):
        z = safe_sparse_dot(self.A, x, dense_output=True).ravel() - self.b
        pen = self.alpha * safe_sparse_dot(x.T, x, dense_output=True).ravel()[0]
        loss = 0.5 * (z * z).mean() + 0.5 * pen
        if not return_gradient:
            return loss
        grad = safe_sparse_add(self.A.T.dot(z) / self.A.shape[0], self.alpha * x.T)
        return loss, np.asarray(grad).ravel()

    @property
    def lipschitz(self):
        s = splinalg.svds(self.A, k=1, return_singular_vectors=False)[0]
        return (s * s) / self.A.shape[0] + self.alpha


class HuberLoss:
    """Huber loss"""

    def __init__(self, A, b, alpha=0, delta=1):
        self.delta = delta
        self.A = A
        self.b = b
        self.alpha = alpha
        self.name = "huber"

    def __call__(self, x):
        return self.f_grad(x, return_gradient=False)

    def f_grad(self, x, return_gradient=True):
        z = safe_sparse_dot(self.A, x, dense_output=True).ravel() - self.b
        idx = np.abs(z) < self.delta
        loss = 0.5 * np.sum(z[idx] * z[idx])
        loss += np.sum(self.delta * (np.abs(z[~idx]) - 0.5 * self.delta))
        loss = (
            loss / z.size
            + 0.5 * self.alpha * safe_sparse_dot(x.T, x, dense_output=True).ravel()[0]
        )
        if not return_gradient:
            return loss
        grad = self.A[idx].T.dot(z[idx]) / self.A.shape[0] + self.alpha * x.T
        grad = np.asarray(grad)
        grad += self.A[~idx].T.dot(self.delta * np.sign(z[~idx])) / self.A.shape[0]
        return loss, np.asarray(grad).ravel()

    @property
    def lipschitz(self):
        s = splinalg.svds(self.A, k=1, return_singular_vectors=False)[0]
        return (s * s) / self.A.shape[0] + self.alpha


class L1Norm:
    """L1 norm, that is, the sum of absolute values:

  .. math::
      \\alpha\\sum_i^d |x_i|

  Args:
  alpha: float
      constant multiplying the L1 norm

  """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * np.abs(x).sum()

    def prox(self, x, step_size):
        return np.fmax(x - self.alpha * step_size, 0) - np.fmax(
            -x - self.alpha * step_size, 0
        )

    def prox_factory(self, n_features):
        alpha = self.alpha

        @njit
        def _prox_L1(x, i, indices, indptr, d, step_size):
            for j in range(indptr[i], indptr[i + 1]):
                j_idx = indices[j]  # for L1 this is the same
                a = x[j_idx] - alpha * d[j_idx] * step_size
                b = -x[j_idx] - alpha * d[j_idx] * step_size
                x[j_idx] = np.fmax(a, 0) - np.fmax(b, 0)

        return _prox_L1, sparse.eye(n_features, format="csr")


class L1Ball:
    """Indicator function over the L1 ball

  This function is 0 if the sum of absolute values is less than or equal to
  alpha, and infinity otherwise.
  """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        if np.abs(x).sum() <= self.alpha:
            return 0
        else:
            return np.infty

    def prox(self, x, step_size):
        return euclidean_proj_l1ball(x, self.alpha)

    def lmo(self, u, x):
        """Solve the linear problem
    min_{||s||_1 <= alpha} <u, s>
    """
        abs_u = np.abs(u)
        largest_coordinate = np.argmax(abs_u)

        update_direction = -x.copy()
        update_direction[largest_coordinate] += self.alpha * np.sign(
            u[largest_coordinate]
        )

        return update_direction

    def lmo_pairwise(self, u, x, active_set):
        # XXX do we actually need x?
        if np.any(active_set < 0):
            raise RuntimeError("active set coefficients cannot be negative")

        u2 = np.concatenate((u, -u))
        largest_coordinate = np.argmax(u2)

        u2_active = ma.array(u2, mask=(active_set == 0))
        largest_active = np.argmax(-u2_active)

        update_direction = np.zeros_like(x)
        sign_largest = 1 if largest_coordinate < len(u) else -1
        idx_largest = largest_coordinate - len(u) * (largest_coordinate >= len(u))
        update_direction[idx_largest] = self.alpha * sign_largest

        idx_largest_active = largest_active - len(u) * (largest_active >= len(u))
        sign_active = 1 if largest_active < len(u) else -1
        update_direction[idx_largest_active] -= self.alpha * sign_active

        return update_direction, largest_coordinate, largest_active


class GroupL1:
    """
  Group Lasso penalty

  Parameters
  ----------

  alpha: float
      Constat multiplying this loss

  blocks: list of lists

  """

    def __init__(self, alpha, groups):
        self.alpha = alpha
        # groups need to be increasing
        for i, g in enumerate(groups):
            if not np.all(np.diff(g) == 1):
                raise ValueError("Groups must be contiguous")
            if i > 0 and groups[i - 1][-1] >= g[0]:
                raise ValueError("Groups must be increasing")
        self.groups = groups

    def __call__(self, x):
        return self.alpha * np.sum([np.linalg.norm(x[g]) for g in self.groups])

    def prox(self, x, step_size):
        out = x.copy()
        for g in self.groups:

            norm = np.linalg.norm(x[g])
            if norm > self.alpha * step_size:
                out[g] -= step_size * self.alpha * out[g] / norm
            else:
                out[g] = 0
        return out

    def prox_factory(self, n_features):
        B_data = np.zeros(n_features)
        B_indices = np.arange(n_features, dtype=np.int32)
        B_indptr = np.zeros(n_features + 1, dtype=np.int32)

        feature_pointer = 0
        block_pointer = 0
        for g in self.groups:
            while feature_pointer < g[0]:
                # non-penalized feature
                B_data[feature_pointer] = -1.0
                B_indptr[block_pointer + 1] = B_indptr[block_pointer] + 1
                feature_pointer += 1
                block_pointer += 1
            B_indptr[block_pointer + 1] = B_indptr[block_pointer]
            for _ in g:
                B_data[feature_pointer] = 1.0
                B_indptr[block_pointer + 1] += 1
                feature_pointer += 1
            block_pointer += 1
        for _ in range(feature_pointer, n_features):
            B_data[feature_pointer] = -1.0
            B_indptr[block_pointer + 1] = B_indptr[block_pointer] + 1
            feature_pointer += 1
            block_pointer += 1

        B_indptr = B_indptr[: block_pointer + 1]
        B = sparse.csr_matrix((B_data, B_indices, B_indptr))
        alpha = self.alpha

        @njit
        def _prox_gl(x, i, indices, indptr, d, step_size):
            for b in range(indptr[i], indptr[i + 1]):
                h = indices[b]
                if B_data[B_indices[B_indptr[h]]] <= 0:
                    continue
                ss = step_size * d[h]
                norm = 0.0
                for j in range(B_indptr[h], B_indptr[h + 1]):
                    j_idx = B_indices[j]
                    norm += x[j_idx] ** 2
                norm = np.sqrt(norm)
                if norm > alpha * ss:
                    for j in range(B_indptr[h], B_indptr[h + 1]):
                        j_idx = B_indices[j]
                        x[j_idx] *= 1 - alpha * ss / norm
                else:
                    for j in range(B_indptr[h], B_indptr[h + 1]):
                        j_idx = B_indices[j]
                        x[j_idx] = 0.0

        return _prox_gl, B


class FusedLasso:
    """
  Fused Lasso penalty

  Parameters
  ----------

  alpha: scalar

  Examples
  --------
  """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * np.sum(np.abs(np.diff(x)))

    def prox(self, x, step_size):
        # imported here to avoid circular imports
        from copt import tv_prox

        return tv_prox.prox_tv1d(x, step_size * self.alpha)

    def prox_1_factory(self, n_features):
        B_1_data = np.ones(n_features)
        B_1_indices = np.arange(n_features, dtype=np.int32)
        B_1_indptr = np.arange(0, n_features + 1, 2, dtype=np.int32)
        if n_features % 2 == 1:
            B_1_indptr = np.concatenate((B_1_indptr, [B_1_indptr[-1] + 1]))
            B_1_data[-1] = -1
        n_blocks = (n_features + 1) // 2
        B_1 = sparse.csr_matrix(
            (B_1_data, B_1_indices, B_1_indptr), shape=(n_blocks, n_features)
        )
        alpha = self.alpha

        @njit
        def _prox_1_fl(x, i, indices, indptr, d, step_size):
            for b in range(indptr[i], indptr[i + 1]):
                h = indices[b]
                j_idx = B_1_indices[B_1_indptr[h]]
                if B_1_data[j_idx] <= 0:
                    continue
                ss = step_size * d[h] * alpha
                if x[j_idx] - ss >= x[j_idx + 1] + ss:
                    x[j_idx] -= ss
                    x[j_idx + 1] += ss
                elif x[j_idx] + ss <= x[j_idx + 1] - ss:
                    x[j_idx] += ss
                    x[j_idx + 1] -= ss
                else:
                    avg = (x[j_idx] + x[j_idx + 1]) / 2.0
                    x[j_idx] = avg
                    x[j_idx + 1] = avg

        return _prox_1_fl, B_1

    def prox_2_factory(self, n_features):
        B_2_data = np.ones(n_features)
        B_2_indices = np.arange(n_features, dtype=np.int32)
        _indptr = np.arange(1, n_features + 2, 2, dtype=np.int32)
        B_2_indptr = np.concatenate(([0], _indptr))
        B_2_data[0] = -1
        if n_features % 2 == 0:
            B_2_indptr[-1] -= 1
            B_2_data[-1] = -1
        n_blocks = n_features // 2 + 1
        B_2 = sparse.csr_matrix(
            (B_2_data, B_2_indices, B_2_indptr), shape=(n_blocks, n_features)
        )
        alpha = self.alpha

        @njit
        def _prox_2_fl(x, i, indices, indptr, d, step_size):
            for b in range(indptr[i], indptr[i + 1]):
                h = indices[b]
                j_idx = B_2_indices[B_2_indptr[h]]
                if B_2_data[j_idx] <= 0:
                    continue
                ss = step_size * d[h] * alpha
                if x[j_idx] - ss >= x[j_idx + 1] + ss:
                    x[j_idx] -= ss
                    x[j_idx + 1] += ss
                elif x[j_idx] + ss <= x[j_idx + 1] - ss:
                    x[j_idx] += ss
                    x[j_idx + 1] -= ss
                else:
                    avg = (x[j_idx] + x[j_idx + 1]) / 2.0
                    x[j_idx] = avg
                    x[j_idx + 1] = avg

        return _prox_2_fl, B_2


class SimplexConstraint:
    def __init__(self, s=1):
        self.s = s

    def prox(self, x, step_size):
        return euclidean_proj_simplex(x, self.s)


def euclidean_proj_simplex(v, s=1.0):
    r""" Compute the Euclidean projection on a positive simplex
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
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
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
    if len(v.shape) > 1:
        raise ValueError
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
        s_threshold = np.fmax(s - self.alpha * step_size, 0) - np.fmax(
            -s - self.alpha * step_size, 0
        )
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

    def lmo(self, u, x):
        u_mat = u.reshape(self.shape)
        ut, _, vt = splinalg.svds(u_mat, k=1)
        vertex = self.alpha * np.outer(ut, vt).ravel()
        update_direction = vertex - x
        return update_direction


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
        # here to avoid circular imports
        from copt import tv_prox

        return tv_prox.prox_tv2d(
            x,
            step_size * self.alpha,
            self.n_rows,
            self.n_cols,
            max_iter=self.max_iter,
            tol=self.tol,
        )
