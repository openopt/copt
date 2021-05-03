import numpy as np
from numpy import ma as ma
from scipy import linalg
from scipy.sparse import linalg as splinalg

class LinfBall:
    """L-infinity ball.
    
    Args:
        alpha: float
            radius of the ball.
    """
    p = np.inf

    def __init__(self, alpha):
        self.alpha = alpha

    def prox(self, x, step_size=None):
        """Projection onto the L-infinity ball.
        
        Args:
            x: array-like

        Returns:
            p : array-like, same shape as x
                projection of x onto the L-infinity ball.
        """
        return x.clip(-self.alpha, self.alpha)


class L2Ball:
    """L2 ball.
    
    Args:
        alpha: float
            radius of the ball.
    """
    p = 2

    def __init__(self, alpha):
        self.alpha = alpha

    def prox(self, x, step_size=None):
        """Projection onto the L-2 ball.
        
        Args:
            x: array-like

        Returns:
            p : array-like, same shape as x
                projection of x onto the L-2 ball.
        """

        norm = np.sqrt((x ** 2).sum())
        if norm <= self.alpha:
            return x
        return self.alpha * x / norm


class L1Ball:
    """Indicator function over the L1 ball

    This function is 0 if the sum of absolute values is less than or equal to
    alpha, and infinity otherwise.

    Args:
        alpha: float
            radius of the ball.
    """
    p = 1

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        if np.abs(x).sum() <= self.alpha:
            return 0
        else:
            return np.infty

    def prox(self, x, step_size=None):
        """Projection onto the L-infinity ball.
        
        Parameters
        ----------
        x: array-like

        Returns
        -------
        p : array-like, same shape as x
            projection of x onto the L-infinity ball.
        """
        return euclidean_proj_l1ball(x, self.alpha)

    def lmo(self, u, x, active_set=None):
        """Linear Minimization Oracle.

        Return s - x with s solving the linear problem
            max_{||s||_1 <= alpha} <u, s>

        Args:
          u: array-like
              usually -gradient
          x: array-like
              usually the iterate of the considered algorithm
          active_set: no effect here.

        Returns:
          update_direction: array,
              s - x, where s is the vertex of the constraint most correlated
              with u
          fw_vertex_rep: (float, int)
              a hashable representation of s, for active set management
          None: not used here
          max_step_size: float
              1. for a Frank-Wolfe step.
        """
        abs_u = np.abs(u)
        largest_coordinate = np.argmax(abs_u)
        sign = np.sign(u[largest_coordinate])

        update_direction = -x.copy()
        update_direction[largest_coordinate] += self.alpha * sign

        # Only useful for active_set management in pairwise FW
        fw_vertex_rep = (sign, largest_coordinate)
        max_step_size = 1.
        return update_direction, fw_vertex_rep, None, max_step_size

    def lmo_pairwise(self, u, x, active_set):
        """Pairwise Linear Minimization Oracle.
        
        Return s - v with s solving the linear problem
            max_{||s||_1 <= alpha} <u, s>
        and v solving the linear problem
            min_{v \in active_set} <u, s>
        
        Args:
          u: array,
              usually -gradient
          x: array,
              usually the iterate of the considered algorithm
          active_set: used to compute v
          
        Returns:
          update_direction: array
              s - v, where s is the vertex of the constraint most correlated with u
              and v is the vertex of the active set least correlated with u
          fw_vertex_rep: (float, int)
              a hashable representation of s, for active set management
          away_vertex_rep: (float, int)
              a hashable representation of v, for active set management
          max_step_size: float
              max_step_size to not move out of the constraint. Given by active_set[away_vertex_rep].
        """
        update_direction, fw_vertex_rep, _,  _ = self.lmo(u, x)
        update_direction += x

        def _correlation(vertex_rep, u):
            """Compute the correlation between vertex represented by vertex_rep and vector u."""
            sign, idx = vertex_rep
            return sign * u[idx]

        away_vertex_rep, max_step_size = min(active_set.items(),
                                             key=lambda item: _correlation(item[0], u))

        sign, idx = away_vertex_rep
        update_direction[idx] -= sign * self.alpha
        return update_direction, fw_vertex_rep, away_vertex_rep, max_step_size


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

        return update_direction, int(largest_coordinate), None, 1

def euclidean_proj_simplex(v, s=1.0):
    r""" Compute the Euclidean projection on a positive simplex

    Solves the optimization problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0

    Args:
    v: (n,) numpy array,
        n-dimensional vector to project
    s: float, optional, default: 1,
        radius of the simplex

    Returns:
    w: (n,) numpy array,
        Euclidean projection of v on the simplex

    Notes:
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References:
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

    Args:
        v: (n,) numpy array,
            n-dimensional vector to project
        s: float, optional, default: 1,
            radius of the L1-ball

    Returns:
        w: (n,) numpy array,
            Euclidean projection of v on the L1-ball of radius s

    Notes:
        Solves the problem by a reduction to the positive simplex case
        See also :ref:`euclidean_proj_simplex`
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


class TraceBall:
    """Projection onto the trace (aka nuclear) norm, sum of singular values

    Args:
        alpha: float
            radius of the ball.

    """

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

    def lmo(self, u, x, active_set=None):
        """Linear Minimization Oracle.
        
        Return s - x with s solving the linear problem
            max_{||s||_nuc <= alpha} <u, s>
        
        Args:
          u: usually -gradient
          x: usually the iterate of the considered algorithm
          active_set: no effect here.
          
        Returns:
          update_direction: s - x, where s is the vertex of the constraint most correlated with u
          None: not used here
          None: not used here
          max_step_size: 1. for a Frank-Wolfe step.
        """
        u_mat = u.reshape(self.shape)
        ut, _, vt = splinalg.svds(u_mat, k=1)
        vertex = self.alpha * np.outer(ut, vt).ravel()
        return vertex - x, None, None, 1.


class TraceSpectrahedron:
    """Projection of a square, symmetric matrix onto the set of
    positive-semidefinite matrices with bounded trace norm.

    Args:
        alpha: float
            radius of the spectrahedron.
        dim: int
            The ambient space is of dimension dim x dim

    """
    def __init__(self, alpha, dim):
        assert type(dim) == int
        self.alpha = alpha
        self.dim = dim
        self._shape = (dim,dim)

    def __call__(self, x):
        X = x.reshape(self._shape)
        eigvals = linalg.eigvalsh(X)
        # check that all non-zero eigenvalues are greater than zero
        is_psd = np.all(eigvals[~np.isclose(0, eigvals)] > 0)
        is_in_ball = eigvals.sum() <= self.alpha + np.finfo(np.float32).eps
        if is_psd and is_in_ball:
            return 0
        else:
            return np.inf

    def prox(self, x, step_size):
        X = x.reshape(self._shape)
        X = .5*(X + X.T)
        s, U = linalg.eigh(X)
        s_psd = np.maximum(s, 0)
        s_psd_threshold = euclidean_proj_l1ball(s_psd, self.alpha) 
        ret = (U * s_psd_threshold).dot(U.T).ravel()
        assert np.all(ret == ret.T)
        return ret

    def lmo(self, u, x, active_set=None):
        r"""Linear Maximization Oracle.

        Returns s - x with s solving the following
            max_{s\in D} <u,s> 
            where D := {X | X is p.s.d.;  ||X||_nuc <= alpha}

        Args:
          u: usually -gradient
          x: usually the iterate of the considered algorithm
          active_set: ignored

        Returns:
          update_direction: s - x, where s is the vertex of the constraint most correlated with u
          None: not used here
          None: not used here
          max_step_size: 1. for a Frank-Wolfe step.
        """
        u_mat = u.reshape(self._shape)
        u_mat = .5*(u_mat + u_mat)
        s, ut = splinalg.eigsh(u_mat, k=1, which='LA')

        if s < 0:
            vertex = np.zeros(self._shape).ravel()
        else:
            vertex = self.alpha * np.outer(ut,ut).ravel()
        return vertex - x, None, None, 1.


class RowEqualityConstraint:
    """Row equality constraint for a matrix-valued decision variable.
    Homotopy smoothing is also implemented.

    Row equality constraints are of the form

    ..math::
        Xv = b

    where we call :math:`v` an operator since it maps the decision variable
    :math:`X` to a vector.

    Homotopy smoothing changes this constraint into a distance of the current
    :math:`X` to the `offset` value, :math:`b`:

    ..math::
        \|Xv - b\|^2

    Args:
      shape: tuple of ints (n,m)
        Describes the underlying of the decision variable x.
      operator: vector of size (m,)
      offset: vector of size (n,)
      beta_scaling: ignored
      name: string, optional
        Used by other codes (e.g. trace objects) to identify this particular
        constraint.
    """
    def __init__(self, shape, operator, offset, beta_scaling=1.0, name='row_equality_constraint'):
        # TODO incorporate beta_scaling.  Currently not done because it is only
        # used by element wise constraints.
        assert len(shape) == 2
        assert len(offset.shape) == 1
        assert len(operator.shape) == 2
        assert operator.shape[0] == shape[1]
        assert offset.shape[0] == shape[0]
        self.shape = shape
        self.operator = operator
        self.offset = offset
        self.name = name
        self.offset_norm = np.linalg.norm(self.offset)

    def __call__(self, x):
        X = x.reshape(self.shape)
        z = np.matmul(X, self.operator)
        if np.all(z == self.offset):
            return 0
        else:
            return np.inf

    def apply_operator(self, x):
        """Evaluates A(x).  In this case, `x * operator` where '*' denotes
        matrix multiplication.
        """
        X = x.reshape(self.shape)
        return X.dot(self.operator)

    def smoothed_grad(self, x):
        """Returns the value and the gradient of the homotopy smoothed
        constraint.

        Args:
          x: decision variable
        """
        X = x.reshape(self.shape)
        err = X.dot(self.operator) - self.offset
        val = np.linalg.norm(err) ** 2
        grad = 2*np.outer(err, self.operator)
        return val, grad.flatten()

    def feasibility(self, x):
        """Returns a normalized distance of the current iterate, x, to the
        constraint specified by the offset. Intended usage is for tracking
        the progress of the algorithm (e.g. by a trace object)

        Args:
          x: decision variable
        """
        X = x.reshape(self.shape)
        err = X.dot(self.operator) - self.offset
        val = np.linalg.norm(err)
        return val / self.offset_norm


class ElementWiseInequalityConstraint:
    """Element-wise inequality constraint for vector or matrix-valued
    decision variables. Homotopy smoothing is also implemented.

    ..math::
        X_{ij} \geq c

    where :math:`c\in\mathbb R`

    Homotopy changes this constraint into a simple distance 

    ..math::
        \|X - c\|^2

    which is used to compute gradients and feasibility

    Args:
      shape: 2-ple.
        Shape of the underlying matrix.
      operator: ignored
      offset: ignored
      beta_scaling: float
        The relative scaling of this constraint with respect to other constraints.
    """
    def __init__(self, shape, operator, offset, beta_scaling=1000.,
                 name='elementwise_inequality_constraint', eps=np.finfo(np.float32).eps):
        assert len(shape) == 2
        assert operator.shape == shape
        self.offset = offset
        self.beta_scaling = beta_scaling
        self.name = name
        self.eps = eps

    def __call__(self, x):
        if np.all(x + self.eps >= 0):
            return 0
        else:
            return np.inf

    def smoothed_grad(self, x):
        """Returns the value and gradient (flattened) of the smoothed
        constraint, i.e. (np.float-like, np.array-like).

        Args:
          x: decision variable
        """
        mask = (x+self.eps>=0)
        grad = x.copy()
        grad[mask] = 0
        val = linalg.norm(grad)**2
        return val, self.beta_scaling*grad

    def feasibility(self, x):
        """Returns the norm of the elements of x which do not satisfy the
        constraint. Elements which satisfy the constraint are not included.
        """
        mask = (x+self.eps>=0)
        grad = x.copy()
        grad[mask] = 0
        return linalg.norm(grad)
