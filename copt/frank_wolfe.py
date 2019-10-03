"""Frank-Wolfe and related algorithms."""
import warnings
import numpy as np
from scipy import linalg
from scipy import optimize
from tqdm import trange
from copt import line_search


def _dr_step_size(lipschitz_t, certificate, norm_update_direction, max_step_size):
    # .. Demyanov-Rubinov step-size ..
    return min(certificate / (norm_update_direction * lipschitz_t), max_step_size)


def _adaptive_step_size_scipy(
    f_grad,
    x,
    f_t,
    grad,
    old_f_t,
    lipschitz_t,
    certificate,
    update_direction,
    norm_update_direction,
    max_step_size,
):
    out = line_search.line_search_wolfe1(
        lambda z: f_grad(z)[0],
        lambda z: f_grad(z)[1],
        x,
        update_direction,
        gfk=grad,
        old_fval=f_t,
        old_old_fval=old_f_t,
        amax=max_step_size,
    )
    step_size_t = out[0]
    f_next = out[3]
    grad_next = out[-1]
    if step_size_t is None:
        tmp = certificate / (norm_update_direction * lipschitz_t)
        step_size_t = min(tmp, max_step_size)
        f_next, grad_next = f_grad(x + step_size_t * update_direction)
    return step_size_t, f_next, grad_next


def _adaptive_step_size_panj(
    f_grad,
    x,
    f_t,
    grad,
    old_f_t,
    lipschitz_t,
    certificate,
    update_direction,
    norm_update_direction,
    max_step_size,
):

    ratio_decrease = 0.9
    sigma = 1.0
    max_iter = 100
    if old_f_t is not None:
        tmp = (certificate ** 2) / (2 * (old_f_t - f_t) * norm_update_direction)
        lipschitz_t = max(min(tmp, lipschitz_t), lipschitz_t * ratio_decrease)
    for _ in range(max_iter):
        step_size_t = min(
            certificate / (norm_update_direction * lipschitz_t), max_step_size
        )
        f_next, grad_next = f_grad(x + step_size_t * update_direction)
        if (f_next - f_t) / step_size_t < -sigma * certificate / 2:
            # we're done here
            break
        else:
            lipschitz_t *= 2.0
    else:
        warnings.warn(
            "Exhausted line search iterations in minimize_frank_wolfe", RuntimeWarning
        )

    return step_size_t, f_next, grad_next, lipschitz_t


def minimize_frank_wolfe(
    f_grad,
    x0,
    lmo,
    step_size="adaptive_scipy",
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
      Takes as input a vector u of same size as x0 and returns a solution to
      the linear minimization oracle (defined above).

    step_size: None or "adaptive" or "adaptive2" or callable
      Step-size step_size to use. If None is used and keyword lipschitz
      is not given or None, then it will use a decreasing step-size of the
      form 2/(k+1) (described in [1]). If None is used and keyword lipschitz
      is not None, then it will use the Demyanov-Rubinov step-size step_size
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
    if lipschitz is not None:
        lipschitz_t = lipschitz

    pbar = trange(max_iter, disable=(verbose == 0))
    f_t, grad = f_grad(x)
    old_f_t = None

    it = 0
    for it in pbar:
        update_direction = lmo(-grad, x)
        certificate = np.dot(update_direction, -grad)

        if certificate <= tol:
            break
        norm_update_direction = linalg.norm(update_direction) ** 2
        if hasattr(step_size, "__call__"):
            step_size_t = step_size(locals())
            f_next, grad_next = f_grad(x + step_size_t * update_direction)
        elif step_size == "adaptive_scipy":
            step_size_t, f_next, grad_next = _adaptive_step_size_scipy(
                f_grad,
                x,
                f_t,
                grad,
                old_f_t,
                lipschitz_t,
                certificate,
                update_direction,
                norm_update_direction,
                1,
            )
        elif step_size == "adaptive":
            step_size_t, f_next, grad_next, lipschitz_t = _adaptive_step_size_panj(
                f_grad,
                x,
                f_t,
                grad,
                old_f_t,
                lipschitz_t,
                certificate,
                update_direction,
                norm_update_direction,
                1,
            )
        elif step_size == "DR":
            if lipschitz is None:
                raise ValueError('lipschitz needs to be specified with step_size="DR"')
            step_size_t = _dr_step_size(
                lipschitz_t, certificate, norm_update_direction, 1
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
        pbar.set_postfix(tol=certificate, iter=it, L_t=lipschitz_t)

        old_f_t = f_t
        f_t, grad = f_next, grad_next
    if callback is not None:
        callback(locals())
    pbar.close()
    return optimize.OptimizeResult(x=x, nit=it, certificate=certificate)


def minimize_pairwise_frank_wolfe(
    f_grad,
    x0,
    active_set,
    lmo_pairwise,
    step_size=None,
    lipschitz=None,
    max_iter=200,
    tol=1e-12,
    callback=None,
    verbose=0,
):
    """Pairwise FW on the L1 ball.

.. warning::
    This feature is experimental, API is likely to change.


  Design: LMO takes as input the active set (an array of shape
  n_vertices).

  should return d_t, certificate, and both indices of the selected vertices.

  How to pass the initialization of vertices?

  :Args:
    f_grad: callable
      Takes as input the current iterate (a vector of same size as x0) and
      returns the function value and gradient of the objective function.
      It should accept the optional argument return_gradient, and when False
      it should return only the function value.

    x0: array-like
      Initial guess for solution.

    lmo_pairwise: callable
      Takes as input a vector u of same size as x0 and returns a solution to
      the pairwise linear minimization oracle (see doc).

    step_size: None or "adaptive" or "adaptive2" or callable
      Step-size step_size to use. If None is used and keyword lipschitz
      is not given or None, then it will use a decreasing step-size of the
      form 2/(k+1) (described in [1]). If None is used and keyword lipschitz
      is not None, then it will use the Demyanov-Rubinov step-size step_size
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


  Returns:
    res : scipy.optimize.OptimizeResult
      The optimization result represented as a
      ``scipy.optimize.OptimizeResult`` object. Important attributes are:
      ``x`` the solution array, ``success`` a Boolean flag indicating if
      the optimizer exited successfully and ``message`` which describes
      the cause of the termination. See `scipy.optimize.OptimizeResult`
      for a description of other attributes.

  Examples:
    * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_sparse_benchmark_pairwise.py`
  """
    x0 = np.asanyarray(x0, dtype=np.float)
    if tol < 0:
        raise ValueError("Tol must be non-negative")
    x = x0.copy()
    lipschitz_t = None
    if lipschitz is not None:
        lipschitz_t = lipschitz
    # .. check active_set ..
    if np.any(active_set < 0):
        raise ValueError("active_set cannot contain negative entries")
    if np.all(active_set == 0):
        raise ValueError("active_set cannot be only zero")

    pbar = trange(max_iter, disable=(verbose == 0))
    f_t, grad = f_grad(x)
    old_f_t = None

    it = 0
    for it in pbar:
        update_direction, idx_s, idx_v = lmo_pairwise(-grad, x, active_set)

        norm_update_direction = linalg.norm(update_direction) ** 2
        certificate = np.dot(update_direction, -grad)

        # .. compute an estimate for the Lipschitz constant ...
        if lipschitz_t is None:
            eps = 1e-3
            grad_eps = f_grad(x + eps * update_direction)[1]
            lipschitz_t = linalg.norm(grad - grad_eps) / (
                eps * np.sqrt(norm_update_direction)
            )
            print("Estimated L_t = %s" % lipschitz_t)

        # compute gamma_max
        max_step_size = active_set[idx_v]

        if certificate <= tol:
            break
        if hasattr(step_size, "__call__"):
            step_size_t = step_size(locals())
            f_next, grad_next = f_grad(x + step_size_t * update_direction)
        elif step_size == "adaptive_scipy":
            step_size_t, f_next, grad_next = _adaptive_step_size_scipy(
                f_grad,
                x,
                f_t,
                grad,
                old_f_t,
                lipschitz_t,
                certificate,
                update_direction,
                norm_update_direction,
                max_step_size,
            )
            assert f_next == f_grad(x + step_size_t * update_direction)[0]
            if step_size_t is None:
                raise RuntimeError
            assert step_size_t >= 0
            assert step_size_t <= max_step_size
        elif step_size == "adaptive":
            step_size_t, f_next, grad_next, lipschitz_t = _adaptive_step_size_panj(
                f_grad,
                x,
                f_t,
                grad,
                old_f_t,
                lipschitz_t,
                certificate,
                update_direction,
                norm_update_direction,
                max_step_size,
            )
        elif step_size == "DR":
            # .. Demyanov-Rubinov step-size ..
            if lipschitz is None:
                raise ValueError('lipschitz needs to be specified with step_size="DR"')
            step_size_t = _dr_step_size(
                lipschitz_t, certificate, norm_update_direction, max_step_size
            )
            f_next, grad_next = f_grad(x + step_size_t * update_direction)
        else:
            raise ValueError("Invalid option step_size=%s" % step_size)
        if callback is not None:
            callback(locals())
        x += step_size_t * update_direction
        active_set[idx_s] += step_size_t
        active_set[idx_v] -= step_size_t
        pbar.set_postfix(tol=certificate, iter=it, L_t=lipschitz_t)

        old_f_t = f_t
        f_t, grad = f_next, grad_next
    if callback is not None:
        callback(locals())
    pbar.close()
    return optimize.OptimizeResult(x=x, nit=it, certificate=certificate)


def minimize_away_frank_wolfe(
    f_grad,
    x0,
    active_set,
    lmo_pairwise,
    step_size=None,
    lipschitz=None,
    max_iter=200,
    tol=1e-12,
    callback=None,
    verbose=0,
):
    """Pairwise FW on the L1 ball.

.. warning::
    This feature is experimental, API is likely to change.


  Design: LMO takes as input the active set (an array of shape
  n_vertices).

  should return d_t, certificate, and both indices of the selected vertices.

  How to pass the initialization of vertices?

  :Args:
    f_grad: callable
      Takes as input the current iterate (a vector of same size as x0) and
      returns the function value and gradient of the objective function.
      It should accept the optional argument return_gradient, and when False
      it should return only the function value.

    x0: array-like
      Initial guess for solution.

    lmo_pairwise: callable
      Takes as input a vector u of same size as x0 and returns a solution to
      the pairwise linear minimization oracle (see doc).

    step_size: None or "adaptive" or "adaptive2" or callable
      Step-size step_size to use. If None is used and keyword lipschitz
      is not given or None, then it will use a decreasing step-size of the
      form 2/(k+1) (described in [1]). If None is used and keyword lipschitz
      is not None, then it will use the Demyanov-Rubinov step-size step_size
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


  Returns:
    res : scipy.optimize.OptimizeResult
      The optimization result represented as a
      ``scipy.optimize.OptimizeResult`` object. Important attributes are:
      ``x`` the solution array, ``success`` a Boolean flag indicating if
      the optimizer exited successfully and ``message`` which describes
      the cause of the termination. See `scipy.optimize.OptimizeResult`
      for a description of other attributes.

  Examples:
    * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_sparse_benchmark_pairwise.py`
  """
    x0 = np.asanyarray(x0, dtype=np.float)
    if tol < 0:
        raise ValueError("Tol must be non-negative")
    x = x0.copy()
    lipschitz_t = None
    if lipschitz is not None:
        lipschitz_t = lipschitz
    # .. check active_set ..
    if np.any(active_set < 0):
        raise ValueError("active_set cannot contain negative entries")
    if np.all(active_set == 0):
        raise ValueError("active_set cannot be only zero")

    pbar = trange(max_iter, disable=(verbose == 0))
    f_t, grad = f_grad(x)
    old_f_t = None

    it = 0
    for it in pbar:
        update_direction, idx_s, idx_v = lmo_pairwise(-grad, x, active_set)

        norm_update_direction = linalg.norm(update_direction) ** 2
        certificate = np.dot(update_direction, -grad)

        # .. compute an estimate for the Lipschitz constant ...
        if lipschitz_t is None:
            eps = 1e-3
            grad_eps = f_grad(x + eps * update_direction)[1]
            lipschitz_t = linalg.norm(grad - grad_eps) / (
                eps * np.sqrt(norm_update_direction)
            )

        # compute gamma_max
        max_step_size = active_set[idx_v]

        if certificate <= tol:
            break
        if hasattr(step_size, "__call__"):
            step_size_t = step_size(locals())
            f_next, grad_next = f_grad(x + step_size_t * update_direction)
        elif step_size == "adaptive_scipy":
            step_size_t, f_next, grad_next = _adaptive_step_size_scipy(
                f_grad,
                x,
                f_t,
                grad,
                old_f_t,
                lipschitz_t,
                certificate,
                update_direction,
                norm_update_direction,
                max_step_size,
            )
            assert f_next == f_grad(x + step_size_t * update_direction)[0]
            if step_size_t is None:
                raise RuntimeError
            assert step_size_t >= 0
            assert step_size_t <= max_step_size
        elif step_size == "adaptive":
            step_size_t, f_next, grad_next, lipschitz_t = _adaptive_step_size_panj(
                f_grad,
                x,
                f_t,
                grad,
                old_f_t,
                lipschitz_t,
                certificate,
                update_direction,
                norm_update_direction,
                max_step_size,
            )
        elif step_size == "DR":
            # .. Demyanov-Rubinov step-size ..
            if lipschitz is None:
                raise ValueError('lipschitz needs to be specified with step_size="DR"')
            step_size_t = _dr_step_size(
                lipschitz_t, certificate, norm_update_direction, max_step_size
            )
            f_next, grad_next = f_grad(x + step_size_t * update_direction)
        else:
            raise ValueError("Invalid option step_size=%s" % step_size)
        if callback is not None:
            callback(locals())
        x += step_size_t * update_direction
        active_set[idx_s] += step_size_t
        active_set[idx_v] -= step_size_t
        pbar.set_postfix(tol=certificate, iter=it, L_t=lipschitz_t)

        old_f_t = f_t
        f_t, grad = f_next, grad_next
    if callback is not None:
        callback(locals())
    pbar.close()
    return optimize.OptimizeResult(x=x, nit=it, certificate=certificate)
