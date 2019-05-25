"""Proximal-gradient algorithms."""
import warnings
import numpy as np
from scipy import optimize
from tqdm import trange


def minimize_proximal_gradient(
    f_grad,
    x0,
    prox=None,
    tol=1e-6,
    max_iter=500,
    verbose=0,
    callback=None,
    step_size="adaptive",
    accelerated=False,
    max_iter_backtracking=1000,
    backtracking_factor=0.6,
):
  """Proximal gradient descent.

  Solves problems of the form

          minimize_x f(x) + g(x)

  where we have access to the gradient of f and the proximal operator of g.

  Args:
    f_grad : callable.
        Value and gradient of f: ``f_grad(x) -> float, array-like``.

    x0 : array-like of size n_features
        Initial guess of solution.

    prox : callable, optional.
        Proximal operator g.

    tol: float

    max_iter : int, optional.
        Maximum number of iterations.

    verbose : int, optional.
        Verbosity level, from 0 (no output) to 2 (output on each iteration)

    callback : callable.
        callback function (optional). Takes a single argument (x) with the
        current coefficients in the algorithm. The algorithm will exit if
        callback returns False.

    step_size : float or "adaptive" or (float, "adaptive").
        Step-size value and/or strategy.

    accelerated: boolean
        Whether to use the accelerated variant of the algorithm.

    max_iter_backtracking: int

    backtracking_factor: float

  Returns:
    res : The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.

  References:
    Beck, Amir, and Marc Teboulle. "Gradient-based algorithms with applications
    to signal recovery." Convex optimization in signal processing and
    communications (2009)

  Examples:
    * :ref:`sphx_glr_auto_examples_plot_group_lasso.py`
  """
  x = x0
  if not max_iter_backtracking > 0:
    raise ValueError("Line search iterations need to be greater than 0")

  if prox is None:
    def prox(x, step_size):
      return x

  if hasattr(step_size, "__len__") and len(step_size) == 2:
    step_size_ = step_size[0]
    step_size = step_size[1]
  else:
    if isinstance(step_size, float):
      step_size_ = step_size
    else:
      # without other information start with a step-size of one
      step_size_ = 1

  success = False
  certificate = np.NaN

  it = 1
  # .. a while loop instead of a for loop ..
  # .. allows for infinite or floating point max_iter ..

  if not accelerated:
    fk, grad_fk = f_grad(x)
    pbar = trange(max_iter, disable=(verbose == 0))
    for it in pbar:
      if callback is not None:
        if callback(locals()) is False:
          break
      # .. compute gradient and step size
      if hasattr(step_size, "__call__"):
        step_size_ = step_size(locals())
        x_next = prox(x - step_size_ * grad_fk, step_size_)
        incr = x_next - x
        f_next, grad_next = f_grad(x_next)
      elif step_size == "adaptive":
        x_next = prox(x - step_size_ * grad_fk, step_size_)
        incr = x_next - x
        step_size_ *= 1.1
        for _ in range(max_iter_backtracking):
          f_next, grad_next = f_grad(x_next)
          rhs = fk + grad_fk.dot(incr) + incr.dot(incr) / (2.0 * step_size_)
          if f_next <= rhs:
            # .. step size found ..
            break
          else:
            # .. backtracking, reduce step size ..
            step_size_ *= backtracking_factor
            x_next = prox(x - step_size_ * grad_fk, step_size_)
            incr = x_next - x
        else:
          warnings.warn("Maxium number of line-search iterations reached")
      else:
        x_next = prox(x - step_size_ * grad_fk, step_size_)
        incr = x_next - x
        f_next, grad_next = f_grad(x_next)
      certificate = np.linalg.norm((x - x_next) / step_size_)
      x[:] = x_next
      fk = f_next
      grad_fk = grad_next

      pbar.set_description("PGD")
      pbar.set_postfix(tol=certificate, step_size=step_size_, iter=it)

      if certificate < tol:
        if verbose:
          pbar.write("Achieved relative tolerance at iteration %s" % it)
        success = True
        break
    else:
      warnings.warn("minimize_proximal_gradient did not reach the desired tolerance level",
                    RuntimeWarning)
  else:
    tk = 1
    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..
    yk = x.copy()
    xk_prev = x.copy()
    pbar = trange(max_iter, disable=(verbose == 0))
    for it in pbar:
      if callback is not None:
        if callback(locals()) is False:
          break

      # .. compute gradient and step size
      current_step_size = step_size_
      grad_fk = f_grad(yk)[1]
      x = prox(yk - current_step_size * grad_fk, current_step_size)
      if step_size == "adaptive":
        for _ in range(max_iter_backtracking):
          incr = x - yk
          if f_grad(x)[0] <= f_grad(yk)[0] + grad_fk.dot(
              incr) + incr.dot(incr) / (2.0 * current_step_size):
            # .. step size found ..
            break
          else:
            # .. backtracking, reduce step size ..
            current_step_size *= backtracking_factor
            x = prox(yk - current_step_size * grad_fk, current_step_size)
        else:
          warnings.warn("Maxium number of line-search iterations reached")
      t_next = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
      yk = x + ((tk - 1.) / t_next) * (x - xk_prev)

      x_prox = prox(x - current_step_size * f_grad(x)[1], current_step_size)
      certificate = np.linalg.norm((x - x_prox) / step_size_)
      tk = t_next
      xk_prev = x.copy()

      if verbose > 0:
        print("Iteration %s, certificate: %s, step size: %s" %
              (it, certificate, step_size_))

      if False and certificate < tol:
        if verbose:
          print("Achieved relative tolerance at iteration %s" % it)
        success = True
        break

      it += 1
    if it >= max_iter:
      warnings.warn("minimize_proximal_gradient did not reach the desired tolerance level",
                    RuntimeWarning)

  pbar.close()
  return optimize.OptimizeResult(
      x=x,
      success=success,
      certificate=certificate,
      nit=it,
      step_size=step_size)

