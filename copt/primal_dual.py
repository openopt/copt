import warnings
import numpy as np
from scipy import optimize
from scipy import linalg


def condat_vu(f_prime, g_prox, h_prox, L, y0, alpha=1.0, beta=1.0, tol=1e-6, max_iter=1000,
              verbose=0, callback=None,
              gamma=1., sigma=1.0, max_iter_ls=20, g_prox_args=(), h_prox_args=()):
    """
    proximal gradient-descent solver for optimization problems of the form

                       minimize_x f(x) + alpha * g(x) + beta * h(L x)

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

    y0 : array-like
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
    yk = np.array(y0, copy=True)
    xk = yk.copy()
    success = False
    if not max_iter_ls > 0:
        raise ValueError('Line search iterations need to be greater than 0')

    # .. main iteration ..
    for it in range(max_iter):

        grad_fk = f_prime(xk)
        x_next = g_prox(xk - gamma * grad_fk - gamma * L.T.dot(xk), gamma * alpha, *g_prox_args)
        z = yk + sigma * L.dot(2 * x_next - xk)
        y_next = z - sigma * h_prox(z, beta / sigma, *h_prox_args)

        yk = y_next
        xk = x_next

        norm_increment = linalg.norm(xk - yk)
        if verbose > 0:
            print("Iteration %s, prox-grad norm: %s" % (it, norm_increment))

        if callback is not None:
            callback(xk)
    else:
        warnings.warn(
            "fmin_cgprox did not reach the desired tolerance level",
            RuntimeWarning)

    return optimize.OptimizeResult(
        x=yk, success=success,
        nit=it)
