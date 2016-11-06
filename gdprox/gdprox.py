import warnings
import numpy as np
from scipy import optimize
from scipy import linalg


def fmin_prox_gd(f, f_prime, g_prox, x0, tol=1e-6, max_iter=1000,
                 verbose=0, callback=None, backtracking=True,
                 step_size=1., max_iter_ls=20):
    """
    proximal gradient-descent solver for optimization problems of the form

                       minimize_x f(x) + g(x)

    where f is a smooth function and g is a (possibly non-smooth)
    function for which the proximal operator is known.

    Parameters
    ----------
    f : callable
        f(x) returns the value of f at x.

    f_prime : callable
        f_prime(x) returns the gradient of f.

    g_prox : callable of the form g_prox(x, alpha)
        g_prox(x, alpha) returns the proximal operator of g at x
        with parameter alpha.

    x0 : array-like
        Initial guess

    backtracking : 'line-search' or float
        XXX Step size.

    max_iter : int
        Maximum number of iterations.

    verbose : int
        Verbosity level, from 0 (no output) to 2 (output on each iteration)

    current_step_size : float
        Starting value for the line-search procedure.

    callback : callable
        callback function (optional).

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.

    References
    ----------
    TODO
    """
    xk = x0
    success = False
    if not max_iter_ls > 0:
        raise ValueError('Line search iterations need to be greater than 0')

    for it in range(max_iter):
        # .. step 1 ..
        # Find suitable step size
        current_step_size = step_size
        grad_fk = f_prime(xk)
        x_next = g_prox(xk - current_step_size * grad_fk, current_step_size)
        incr = x_next - xk
        if backtracking:
            fk = f(xk)
            f_next = f(x_next)
            for _ in range(max_iter_ls):
                if f_next <= fk + grad_fk.dot(incr) + incr.dot(incr) / (2.0 * current_step_size):
                    # step size found
                    break
                else:
                    # backtrack, reduce step size
                    current_step_size *= .4
                    x_next = g_prox(xk - current_step_size * grad_fk, current_step_size)
                    incr = x_next - xk
                    f_next = f(x_next)
        xk = x_next

        norm_increment = linalg.norm(incr, np.inf)
        if verbose > 0:
            print("Iteration %s, prox-grad norm: %s" % (it, norm_increment / current_step_size))

        if norm_increment < tol * backtracking:
            if verbose:
                print("Achieved relative tolerance at iteration %s" % it)
                success = True
            break

        if callback is not None:
            callback(xk)
    else:
        warnings.warn(
            "fmin_cgprox did not reach the desired tolerance level",
            RuntimeWarning)

    return optimize.OptimizeResult(
        x=xk, success=success, fun=fk,
        jac=incr / backtracking,  # prox-grad mapping
        nit=it)
