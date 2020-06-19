import numpy as np
from scipy import sparse, special
from scipy.sparse import linalg as splinalg
from sklearn.utils.extmath import safe_sparse_dot

from copt.utils import safe_sparse_add, njit, prange


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

    def hessian_mv(self, x):
        """Return a callable that returns matrix-vector products with the Hessian."""

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

    def hessian_trace(self, x):
        """Return a callable that returns matrix-vector products with the Hessian."""

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
        @njit(parallel=True)
        def log_deriv(p, y):
            # derivative of logistic loss
            # same as in lightning (with minus sign)
            out = np.zeros_like(p)
            for i in prange(p.size):
                if p[i] < 0:
                    exp_p = np.exp(p[i])
                    out[i] = ((1 - y[i]) * exp_p - y[i]) / (1 + exp_p)
                else:
                    exp_nx = np.exp(-p[i])
                    out[i] = ((1 - y[i]) - y[i] * exp_nx) / (1 + exp_nx)
            return out

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
      \frac{1}{2n}\|A x - b\|^2 + \frac{1}{2} \alpha \|x\|^2

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
    def partial_deriv(self):
        @njit
        def square_deriv(p, y):
            return p - y
        return square_deriv

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