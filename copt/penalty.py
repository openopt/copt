import numpy as np
from scipy import sparse, linalg

from copt.utils import njit


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
        """Proximal operator of the L1 norm.

        This routine can be used in gradient-based methods like
        minimize_proximal_gradient, minimize_three_split and
        minimize_primal_dual.
        """
        return np.fmax(x - self.alpha * step_size, 0) - np.fmax(
            -x - self.alpha * step_size, 0
        )

    def prox_factory(self, n_features):
        """Proximal operator of the L1 norm.

        This method is meant to be used with stochastic algorithms that need
        access to a proximal operator over a potentially sparse vector,
        like minimize_saga, minimize_svrg and minimize_vrtos
        """
        alpha = self.alpha

        @njit
        def _prox_L1(x, i, indices, indptr, d, step_size):
            for j in range(indptr[i], indptr[i + 1]):
                j_idx = indices[j]  # for L1 this is the same
                a = x[j_idx] - alpha * d[j_idx] * step_size
                b = -x[j_idx] - alpha * d[j_idx] * step_size
                x[j_idx] = np.fmax(a, 0) - np.fmax(b, 0)

        return _prox_L1, sparse.eye(n_features, format="csr")


class GroupL1:
    """
  Group Lasso penalty

  Parameters
  ----------

  alpha: float
      Constant multiplying this loss

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