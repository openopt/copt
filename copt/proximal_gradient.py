# python3
"""Proximal-gradient algorithms."""
import warnings
import numpy as np
from scipy import optimize
from copt import utils


def minimize_proximal_gradient(
        fun,
        x0,
        prox=None,
        jac="2-point",
        tol=1e-6,
        max_iter=500,
        args=(),
        verbose=0,
        callback=None,
        step="backtracking",
        accelerated=False,
        eps=1e-8,
        max_iter_backtracking=1000,
        backtracking_factor=0.6,
        trace_certificate=False,
):
    """Proximal gradient descent.

  Solves problems of the form

          minimize_x f(x) + g(x)

  where f is a differentiable function and we have access to the proximal
  operator of g.

  Args:
    fun : callable
        The objective function to be minimized.
            ``fun(x, *args) -> float``
        where x is an 1-D array with shape (n,) and `args`
        is a tuple of the fixed parameters needed to completely
        specify the function.

    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.

    jac : {callable,  '2-point', bool}, optional
        Method for computing the gradient vector. If it is a callable,
        it should be a function that returns the gradient vector:
            ``jac(x, *args) -> array_like, shape (n,)``
        where x is an array with shape (n,) and `args` is a tuple with
        the fixed parameters. Alternatively, the '2-point' select a finite
        difference scheme for numerical estimation of the gradient.
        If `jac` is a Boolean and is True, `fun` is assumed to return the
        gradient along with the objective function. If False, the gradient
        will be estimated using '2-point' finite difference estimation.

    prox : callable, optional.
        Proximal operator g.

    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).

    tol: float, optional
        Tolerance of the optimization procedure. The iteration stops when the gradient mapping
        (a generalization of the gradient to non-smooth functions) is below this tolerance.

    max_iter : int, optional.
        Maximum number of iterations.

    verbose : int, optional.
        Verbosity level, from 0 (no output) to 2 (output on each iteration)

    callback : callable.
        callback function (optional). Takes a single argument (x) with the
        current coefficients in the algorithm. The algorithm will exit if
        callback returns False.

    step : "backtracking" or callable.
        Step-size strategy to use. "backtracking" will use a backtracking line-search,
        while callable will use the value returned by step(locals()).

    accelerated: boolean
        Whether to use the accelerated variant of the algorithm.

    eps: float or ndarray
        If jac is approximated, use this value for the step size.

    max_iter_backtracking: int

    backtracking_factor: float

    trace_certificate: bool

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
    x = np.asarray(x0).flatten()
    if max_iter_backtracking <= 0:
        raise ValueError("Line search iterations need to be greater than 0")

    if prox is None:

        def _prox(x, _):
            return x

        prox = _prox

    success = False
    certificate = np.NaN

    func_and_grad = utils.build_func_grad(jac, fun, args, eps)

    # find initial step-size
    if step == "backtracking":
        step_size = 1.8 / utils.init_lipschitz(func_and_grad, x0)
    else:
        # to avoid step_size being undefined upon return
        step_size = None

    n_iterations = 0
    certificate_list = []
    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..
    if not accelerated:
        fk, grad_fk = func_and_grad(x)
        while True:
            if callback is not None:
                if callback(locals()) is False:  # pylint: disable=g-bool-id-comparison
                    break
            # .. compute gradient and step size
            if hasattr(step, "__call__"):
                step_size = step(locals())
                x_next = prox(x - step_size * grad_fk, step_size)
                update_direction = x_next - x
                f_next, grad_next = func_and_grad(x_next)
            elif step == "backtracking":
                x_next = prox(x - step_size * grad_fk, step_size)
                update_direction = x_next - x
                step_size *= 1.1
                for _ in range(max_iter_backtracking):
                    f_next, grad_next = func_and_grad(x_next)
                    rhs = (
                        fk
                        + grad_fk.dot(update_direction)
                        + update_direction.dot(update_direction) / (2.0 * step_size)
                    )
                    if f_next <= rhs:
                        # .. step size found ..
                        break
                    else:
                        # .. backtracking, reduce step size ..
                        step_size *= backtracking_factor
                        x_next = prox(x - step_size * grad_fk, step_size)
                        update_direction = x_next - x
                else:
                    warnings.warn("Maxium number of line-search iterations reached")
            elif step == "fixed":
                x_next = prox(x - step_size * grad_fk, step_size)
                update_direction = x_next - x
                f_next, grad_next = func_and_grad(x_next)
            else:
                raise ValueError("Step-size strategy not understood")
            certificate = np.linalg.norm((x - x_next) / step_size)
            if trace_certificate:
                certificate_list.append(certificate)
            x[:] = x_next
            fk = f_next
            grad_fk = grad_next

            if certificate < tol:
                success = True
                break

            if n_iterations >= max_iter:
                break
            else:
                n_iterations += 1
        else:
            warnings.warn(
                "minimize_proximal_gradient did not reach the desired tolerance level",
                RuntimeWarning,
            )
    else:
        tk = 1
        # .. a while loop instead of a for loop ..
        # .. allows for infinite or floating point max_iter ..
        yk = x.copy()
        while True:
            grad_fk = func_and_grad(yk)[1]
            if callback is not None:
                if callback(locals()) is False:  # pylint: disable=g-bool-id-comparison
                    break

            # .. compute gradient and step size
            if hasattr(step, "__call__"):
                current_step_size = step(locals())
                x_next = prox(yk - current_step_size * grad_fk, current_step_size)
                t_next = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
                yk = x_next + ((tk - 1.0) / t_next) * (x_next - x)

                t_next = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
                yk = x_next + ((tk - 1.0) / t_next) * (x_next - x)

                x_prox = prox(
                    x_next - current_step_size * func_and_grad(x_next)[1],
                    current_step_size,
                )
                certificate = np.linalg.norm((x - x_prox) / current_step_size)
                tk = t_next
                x = x_next.copy()

            elif step == "backtracking":
                current_step_size = step_size
                x_next = prox(yk - current_step_size * grad_fk, current_step_size)
                for _ in range(max_iter_backtracking):
                    update_direction = x_next - yk
                    if func_and_grad(x_next)[0] <= func_and_grad(yk)[0] + grad_fk.dot(
                        update_direction
                    ) + update_direction.dot(update_direction) / (
                        2.0 * current_step_size
                    ):
                        # .. step size found ..
                        break
                    else:
                        # .. backtracking, reduce step size ..
                        current_step_size *= backtracking_factor
                        x_next = prox(
                            yk - current_step_size * grad_fk, current_step_size
                        )
                else:
                    warnings.warn("Maxium number of line-search iterations reached")
                t_next = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
                yk = x_next + ((tk - 1.0) / t_next) * (x_next - x)

                x_prox = prox(
                    x_next - current_step_size * func_and_grad(x_next)[1],
                    current_step_size,
                )
                certificate = np.linalg.norm((x - x_prox) / current_step_size)
                if trace_certificate:
                    certificate_list.append(certificate)
                tk = t_next
                x = x_next.copy()

            if certificate < tol:
                success = True
                break

            if n_iterations >= max_iter:
                break
            else:
                n_iterations += 1

        if n_iterations >= max_iter:
            warnings.warn(
                "minimize_proximal_gradient did not reach the desired tolerance level",
                RuntimeWarning,
            )

    return optimize.OptimizeResult(
        x=x,
        success=success,
        certificate=certificate,
        nit=n_iterations,
        step_size=step_size,
        trace_certificate=certificate_list,
    )

