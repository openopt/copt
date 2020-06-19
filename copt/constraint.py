import numpy as np
from numpy import ma as ma
from scipy import linalg
from scipy.sparse import linalg as splinalg
from copt.utils import njit


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
        """Return s - x, s solving the linear problem
    max_{||s||_1 <= alpha} <u, s>
    """
        abs_u = np.abs(u)
        largest_coordinate = np.argmax(abs_u)

        update_direction = -x.copy()
        update_direction[largest_coordinate] += self.alpha * np.sign(
            u[largest_coordinate]
        )

        return update_direction, 1

    def lmo_pairwise(self, u, x):
        abs_u = np.abs(u)
        largest_coordinate = np.argmax(abs_u)

        update_direction = np.zeros_like(u)
        update_direction[largest_coordinate] += self.alpha * np.sign(
            u[largest_coordinate]
        )

        u_active = -u * np.sign(x)
        ma_u_active = ma.array(u_active, mask=(u_active == 0))
        largest_active = np.argmax(ma_u_active)
        if largest_active == largest_coordinate:
            # .. if s and v are the same vertex ..
            # .. take a FW step ..
            update_direction -= x
            max_step_size = 1.0
        if u_active[largest_active] > 0:
            update_direction[largest_active] -= self.alpha * np.sign(x[largest_active])
            max_step_size = np.abs(x[largest_active]) / self.alpha
        else:
            # the zero vertex wins
            max_step_size = max(self.alpha - np.sum(np.abs(x)), 0) / self.alpha
            if max_step_size == 0:
                # .. early termination ..
                update_direction[:] = 0

        return update_direction, max_step_size

    # def lmo_away(self, u, x):
    #     raise NotImplementedError
    #     u2 = active_set = np.concatenate((u, -u))
    #     active_set = np.concatenate((x > 0, x < 0))
    #     largest_coordinate = np.argmax(u2)

    #     u2_active = ma.array(u2, mask=(active_set == 0))
    #     largest_active = np.argmax(-u2_active)

    #     update_direction = np.zeros_like(x)
    #     sign_largest = 1 if largest_coordinate < len(u) else -1
    #     idx_largest = largest_coordinate - len(u) * (largest_coordinate >= len(u))
    #     update_direction[idx_largest] = self.alpha * sign_largest

    #     idx_largest_active = largest_active - len(u) * (largest_active >= len(u))
    #     sign_active = 1 if largest_active < len(u) else -1
    #     update_direction[idx_largest_active] -= self.alpha * sign_active

    #     max_step_size = active_set[largest_active]

    #     return update_direction, max_step_size


class SimplexConstraint:
    def __init__(self, s=1):
        self.s = s

    def prox(self, x, step_size):
        return euclidean_proj_simplex(x, self.s)

    def lmo(self, u, x):
        """Return v - x, s solving the linear problem
    max_{||v||_1 <= s, v >= 0} <u, v>
    """
        largest_coordinate = np.argmax(u)

        update_direction = -x.copy()
        update_direction[largest_coordinate] += self.s * np.sign(
            u[largest_coordinate]
        )

        return update_direction, 1


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
    (n,) = v.shape  # will raise ValueError if v is not 1-D
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


class TraceNormBall:
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
        """Return s - x, with s solving the linear problem
    max_{ ||eig(s)||_1 <= alpha } <u, s>
    """
        u_mat = u.reshape(self.shape)
        ut, _, vt = splinalg.svds(u_mat, k=1)
        vertex = self.alpha * np.outer(ut, vt).ravel()
        update_direction = vertex - x
        return update_direction, 1