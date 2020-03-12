"""Frank-Wolfe and related algorithms."""
import warnings
import numpy as np
from scipy import linalg
from scipy import optimize


EPS = np.finfo(np.float32).eps


def adaptive_step_size(
    x,
    f_t,
    old_f_t,
    f_grad,
    certificate,
    lipschitz_t,
    max_step_size,
    update_direction,
    norm_update_direction,
):
    """Adaptive step-size finding routine for FW-like algorithms
    
    Args:
        x: array-like, shape (n_features,)
            Current iterate

        f_t: float
            Value of objective function at the current iterate.

        old_f_t: float
            Value of objective function at previous iterate.

        f_grad: callable
            Callable returning objective function and gradient at
            argument.

        certificate: float
            FW gap

        lipschitz_t: float
            Current value of the Lipschitz estimate.

        max_step_size: float
            Maximum admissible step-size.

        update_direction: array-like, shape (n_features,)
            Update direction given by the FW variant.

        norm_update_direction: float
            Squared L2 norm of update_direction

    Returns:
        step_size_t: float
            Step-size to be used to compute the next iterate.

        lipschitz_t: float
            Updated value for the Lipschitz estimate

        f_next: float
            Objective function evaluated at x + step_size_t d_t.

        grad_next: array-like
            Gradient evaluated at x + step_size_t d_t.
    """
    ratio_decrease = 0.9
    ratio_increase = 2.0
    max_ls_iter = 100
    if old_f_t is not None:
        tmp = (certificate ** 2) / (2 * (old_f_t - f_t) * norm_update_direction)
        lipschitz_t = max(min(tmp, lipschitz_t), lipschitz_t * ratio_decrease)
    for _ in range(max_ls_iter):
        step_size_t = certificate / (norm_update_direction * lipschitz_t)
        if step_size_t < max_step_size:
            rhs = -0.5 * step_size_t * certificate
        else:
            step_size_t = max_step_size
            rhs = (
                -step_size_t * certificate
                + 0.5 * (step_size_t ** 2) * lipschitz_t * norm_update_direction
            )
        f_next, grad_next = f_grad(x + step_size_t * update_direction)
        if f_next - f_t <= rhs + EPS:
            # .. sufficient decrease condition verified ..
            break
        else:
            lipschitz_t *= ratio_increase
    else:
        warnings.warn(
            "Exhausted line search iterations in minimize_frank_wolfe", RuntimeWarning
        )
    return step_size_t, lipschitz_t, f_next, grad_next


def minimize_frank_wolfe(
    f_grad,
    x0,
    lmo,
    step_size="adaptive",
    lipschitz=None,
    max_iter=400,
    tol=1e-12,
    callback=None,
    verbose=0,
):
    r"""Frank-Wolfe algorithm.

  Implements the Frank-Wolfe algorithm, see , see :ref:`frank_wolfe` for
  a more detailed description.

  Args:
    f_grad: callable
      Takes as input the current iterate (a vector of same size as x0) and
      returns the function value and gradient of the objective function.
      It should accept the optional argument return_gradient, and when False
      it should return only the function value.

    x0: array-like
      Initial guess for solution.

    lmo: callable
      Takes as input a vector u of same size as x0 and returns both the update
      direction and the maximum admissible step-size.

    step_size: None or "adaptive" or callable
      Step-size step_size to use. If None is used and keyword lipschitz
      is not given or None, then it will use a decreasing step-size of the
      form 2/(k+2) (described in [1]). If None is used and keyword lipschitz
      is not None, then it will use the Demyanov-Rubinov step-size
      (variant 1 in [2]).

    lipschitz: None or float.
      Estimate for the Lipschitz constant of the gradient.

    max_iter: integer
      Maximum number of iterations.

    tol: float
      Tolerance of the stopping criterion. The algorithm will stop whenever
      the Frank-Wolfe gap is below tol or the maximum number of iterations
      is exceeded.

    callback: callable
      Callback to execute at each iteration. If the callable returns False
      then the algorithm with immediately return.

    verbose: int
      Verbosity level.


  Returns:
    res : scipy.optimize.OptimizeResult
      The optimization result represented as a
      ``scipy.optimize.OptimizeResult`` object. Important attributes are:
      ``x`` the solution array, ``success`` a Boolean flag indicating if
      the optimizer exited successfully and ``message`` which describes
      the cause of the termination. See `scipy.optimize.OptimizeResult`
      for a description of other attributes.


  References:
    [1] Jaggi, Martin. `"Revisiting Frank-Wolfe: Projection-Free Sparse Convex
    Optimization." <http://proceedings.mlr.press/v28/jaggi13-supp.pdf>`_
    ICML 2013.

    [2] Pedregosa, Fabian `"Notes on the Frank-Wolfe Algorithm"
    <http://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/>`_,
    2018

    [3] Pedregosa, Fabian, Armin Askari, Geoffrey Negiar, and Martin Jaggi.
    `"Step-Size Adaptivity in Projection-Free Optimization."
    <https://arxiv.org/pdf/1806.05123.pdf>`_ arXiv:1806.05123 (2018).


  Examples:
    * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_sparse_benchmark.py`
    * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_vertex_overlap.py`
  """
    x0 = np.asanyarray(x0, dtype=np.float)
    if tol < 0:
        raise ValueError("Tol must be non-negative")
    x = x0.copy()
    lipschitz_t = None
    if lipschitz is not None:
        lipschitz_t = lipschitz

    f_t, grad = f_grad(x)
    old_f_t = None

    it = 0
    for it in range(max_iter):
        update_direction, max_step_size = lmo(-grad, x)
        norm_update_direction = linalg.norm(update_direction) ** 2
        certificate = np.dot(update_direction, -grad)

        # .. compute an initial estimate for the ..
        # .. Lipschitz estimate if not given ...
        if lipschitz_t is None:
            eps = 1e-3
            grad_eps = f_grad(x + eps * update_direction)[1]
            lipschitz_t = linalg.norm(grad - grad_eps) / (
                eps * np.sqrt(norm_update_direction)
            )
            print("Estimated L_t = %s" % lipschitz_t)

        if certificate <= tol:
            break
        if hasattr(step_size, "__call__"):
            step_size_t = step_size(locals())
            f_next, grad_next = f_grad(x + step_size_t * update_direction)
        elif step_size == "adaptive":
            step_size_t, lipschitz_t, f_next, grad_next = adaptive_step_size(
                x,
                f_t,
                old_f_t,
                f_grad,
                certificate,
                lipschitz_t,
                max_step_size,
                update_direction,
                norm_update_direction,
            )
        elif step_size == "DR":
            if lipschitz is None:
                raise ValueError('lipschitz needs to be specified with step_size="DR"')
            step_size_t = min(
                certificate / (norm_update_direction * lipschitz_t), max_step_size
            )
            f_next, grad_next = f_grad(x + step_size_t * update_direction)
        elif step_size == "oblivious":
            # .. without knowledge of the Lipschitz constant ..
            # .. we take the oblivious 2/(k+2) step-size ..
            step_size_t = 2.0 / (it + 2)
            f_next, grad_next = f_grad(x + step_size_t * update_direction)
        else:
            raise ValueError("Invalid option step_size=%s" % step_size)
        if callback is not None:
            callback(locals())
        x += step_size_t * update_direction

        old_f_t = f_t
        f_t, grad = f_next, grad_next
    if callback is not None:
        callback(locals())
    return optimize.OptimizeResult(x=x, nit=it, certificate=certificate)
