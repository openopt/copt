import warnings
import numpy as np
from scipy import optimize, linalg, sparse

from . import utils


def minimize_three_split(
    f_grad,
    x0,
    prox_1=None,
    prox_2=None,
    tol=1e-6,
    max_iter=1000,
    verbose=0,
    callback=None,
    line_search=True,
    step_size=None,
    max_iter_backtracking=100,
    backtracking_factor=0.7,
    h_Lipschitz=None,
    args_prox=(),
):
    """Davis-Yin three operator splitting method.

  This algorithm can solve problems of the form

              minimize_x f(x) + g(x) + h(x)

  where f is a smooth function and g and h are (possibly non-smooth)
  functions for which the proximal operator is known.

  Args:
    f_grad: callable
      Returns the function value and gradient of the objective function.
      With return_gradient=False, returns only the function value.

    x0 : array-like
      Initial guess

    prox_1 : callable or None
      prox_1(x, alpha, *args) returns the proximal operator of g at xa
      with parameter alpha.

    prox_2 : callable or None
      prox_2(x, alpha, *args) returns the proximal operator of g at xa
      with parameter alpha.

    tol: float
      Tolerance of the stopping criterion.

    max_iter : int
      Maximum number of iterations.

    verbose : int
      Verbosity level, from 0 (no output) to 2 (output on each iteration)

    callback : callable.
      callback function (optional). Takes a single argument (x) with the
      current coefficients in the algorithm. The algorithm will exit if
      callback returns False.

    line_search : boolean
      Whether to perform line-search to estimate the step size.

    step_size : float
      Starting value for the line-search procedure.

    max_iter_backtracking: int
      maximun number of backtracking iterations.  Used in line search.

    backtracking_factor: float
      the amount to backtrack by during line search.

    args_prox: tuple
      optional Extra arguments passed to the prox functions


  Returns:
    res : OptimizeResult
      The optimization result represented as a
      ``scipy.optimize.OptimizeResult`` object. Important attributes are:
      ``x`` the solution array, ``success`` a Boolean flag indicating if
      the optimizer exited successfully and ``message`` which describes
      the cause of the termination. See `scipy.optimize.OptimizeResult`
      for a description of other attributes.


  References:
    [1] Davis, Damek, and Wotao Yin. `"A three-operator splitting scheme and
    its optimization applications."
    <https://doi.org/10.1007/s11228-017-0421-z>`_ Set-Valued and Variational
    Analysis, 2017.

    [2] Pedregosa, Fabian, and Gauthier Gidel. `"Adaptive Three Operator
    Splitting." <https://arxiv.org/abs/1804.02339>`_ Proceedings of the 35th
    International Conference on Machine Learning, 2018.
  """
    success = False
    if not max_iter_backtracking > 0:
        raise ValueError("Line search iterations need to be greater than 0")

    if prox_1 is None:

        def prox_1(x, s, *args):
            return x

    if prox_2 is None:

        def prox_2(x, s, *args):
            return x

    if step_size is None:
        line_search = True
        step_size = 1.0 / utils.init_lipschitz(f_grad, x0)

    z = prox_2(x0, step_size, *args_prox)
    LS_EPS = np.finfo(np.float).eps

    fk, grad_fk = f_grad(z)
    x = prox_1(z - step_size * grad_fk, step_size, *args_prox)
    u = np.zeros_like(x)

    for it in range(max_iter):

        fk, grad_fk = f_grad(z)
        x = prox_1(z - step_size * (u + grad_fk), step_size, *args_prox)
        incr = x - z
        norm_incr = np.linalg.norm(incr)
        ls = norm_incr > 1e-7 and line_search
        if ls:
            for it_ls in range(max_iter_backtracking):
                rhs = fk + grad_fk.dot(incr) + (norm_incr ** 2) / (2 * step_size)
                ls_tol = f_grad(x, return_gradient=False) - rhs
                if ls_tol <= LS_EPS:
                    # step size found
                    # if ls_tol > 0:
                    #     ls_tol = 0.
                    break
                else:
                    step_size *= backtracking_factor

        z = prox_2(x + step_size * u, step_size, *args)
        u += (x - z) / step_size
        certificate = norm_incr / step_size

        if ls and h_Lipschitz is not None:
            if h_Lipschitz == 0:
                step_size = step_size * 1.02
            else:
                quot = h_Lipschitz ** 2
                tmp = np.sqrt(step_size ** 2 + (2 * step_size / quot) * (-ls_tol))
                step_size = min(tmp, step_size * 1.02)

        if callback is not None:
            if callback(locals()) is False:
                break

        if it > 0 and certificate < tol:
            success = True
            break

    return optimize.OptimizeResult(
        x=x, success=success, nit=it, certificate=certificate, step_size=step_size
    )


def minimize_primal_dual(
    f_grad,
    x0,
    prox_1=None,
    prox_2=None,
    L=None,
    tol=1e-12,
    max_iter=1000,
    callback=None,
    step_size=1.0,
    step_size2=None,
    line_search=True,
    max_iter_ls=20,
    verbose=0,
):
    """Primal-dual hybrid gradient splitting method.

    This method for optimization problems of the form

            minimize_x f(x) + alpha * g(x) + beta * h(L x)

    where f is a smooth function and g is a (possibly non-smooth)
    function for which the proximal operator is known.

    Args:
      f_grad: callable
          Returns the function value and gradient of the objective function.
          It should accept the optional argument return_gradient, and when False
          it should return only the function value.

      prox_1 : callable of the form prox_1(x, alpha)
          prox_1(x, alpha) returns the proximal operator of g at x
          with parameter alpha.

      x0 : array-like
          Initial guess of solution.

      L : ndarray or sparse matrix
          Linear operator inside the h term.

      max_iter : int
          Maximum number of iterations.

      verbose : int
          Verbosity level, from 0 (no output) to 2 (output on each iteration)

      callback : callable.
          callback function (optional). Takes a single argument (x) with the
          current coefficients in the algorithm. The algorithm will exit if
          callback returns False.

    Returns:
      res : OptimizeResult
          The optimization result represented as a
          ``scipy.optimize.OptimizeResult`` object. Important attributes are:
          ``x`` the solution array, ``success`` a Boolean flag indicating if
          the optimizer exited successfully and ``message`` which describes
          the cause of the termination. See `scipy.optimize.OptimizeResult`
          for a description of other attributes.

    References
      Condat, Laurent. "A primal-dual splitting method for convex optimization
      involving Lipschitzian, proximable and linear composite terms." Journal of
      Optimization Theory and Applications (2013).

      Chambolle, Antonin, and Thomas Pock. "On the ergodic convergence rates of a
      first-order primal-dual algorithm." Mathematical Programming (2015)
    """
    x = np.array(x0, copy=True)
    n_features = x.size
    if L is None:
        L = sparse.eye(n_features, n_features, format="csr")

        def Ldot(x):
            return x

        Ltdot = Ldot
    y = L.dot(x)
    success = False
    if not max_iter_ls > 0:
        raise ValueError("Line search iterations need to be greater than 0")

    if prox_1 is None:

        def prox_1(x, step_size):
            return x

    if prox_2 is None:

        def prox_2(x, step_size):
            return x

    # conjugate of prox_2
    def prox_2_conj(x, ss):
        return x - ss * prox_2(x / ss, 1.0 / ss)

    # .. main iteration ..
    theta = 1.0
    delta = 0.99
    sigma = step_size
    tau = step_size2
    if tau is None:
        tau = 0.5 * sigma
    ss_ratio = sigma / tau

    fk, grad_fk = f_grad(x)
    norm_incr = np.infty
    x_next = x.copy()

    for it in range(max_iter):
        y_next = prox_2_conj(y + tau * Ldot(x), tau)
        if line_search:
            tau_next = tau * np.sqrt(1 + theta)
            while True:
                theta = tau_next / tau
                sigma = ss_ratio * tau_next
                y_bar = y_next + theta * (y_next - y)
                x_next = prox_1(x - sigma * (Ltdot(y_bar) + grad_fk), sigma)
                incr_x = np.linalg.norm(Ltdot(x_next) - Ltdot(x))
                if incr_x <= 1e-10:
                    break

                f_next, f_grad_next = f_grad(x_next)
                tmp = (sigma * tau_next) * (incr_x ** 2)
                tmp += 2 * sigma * (f_next - fk - grad_fk.dot(x_next - x))
                if tmp <= delta * (incr_x ** 2) + np.finfo(np.float).eps:
                    tau = tau_next
                    break
                else:
                    tau_next *= 0.5
        else:
            y_bar = 2 * y_next - y
            x_next = prox_1(x - sigma * (Ltdot(y_bar) + grad_fk), sigma)
            f_next, f_grad_next = f_grad(x_next)

        if it % 100 == 0:
            norm_incr = linalg.norm(x_next - x) + linalg.norm(y_next - y)

        x[:] = x_next[:]
        y[:] = y_next[:]
        fk, grad_fk = f_next, f_grad_next

        if norm_incr < tol:
            success = True
            break

        if callback is not None:
            if callback(locals()) is False:
                break

    if it >= max_iter:
        warnings.warn(
            "proximal_gradient did not reach the desired tolerance level",
            RuntimeWarning,
        )

    return optimize.OptimizeResult(x=y, success=success, nit=it, certificate=norm_incr)
