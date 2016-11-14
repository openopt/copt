import warnings
import numpy as np
from scipy import optimize
from scipy import linalg


def primal_dual(f_prime, g_prox, h_prox, L, x0, alpha=1.0, beta=1.0, tol=1e-12,
                max_iter=10000, verbose=0, callback=None, step_size_x=1e-3,
                step_size_y=1e3, max_iter_ls=20, g_prox_args=(), h_prox_args=()):
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

    x0 : array-like
        Initial guess

    L : ndarray or sparse matrix
        Linear operator inside the h term.

    max_iter : int
        Maximum number of iterations.

    verbose : int
        Verbosity level, from 0 (no output) to 2 (output on each iteration)

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
    Chambolle, Antonin, and Thomas Pock. "On the ergodic convergence rates of a
    first-order primal-dual algorithm." Mathematical Programming (2015)
    """
    xk = np.array(x0, copy=True)
    yk = L.dot(xk)
    success = False
    if not max_iter_ls > 0:
        raise ValueError('Line search iterations need to be greater than 0')

    if g_prox is None:
        def g_prox(x, step_size, *args): return x
    if h_prox is None:
        def h_prox(x, step_size, *args): return x

    # conjugate of h_prox
    def h_prox_conj(x, step_size, *args):
        return x - step_size * h_prox(x / step_size,  beta / step_size, *args)
    it = 1
    # .. main iteration ..
    while it < max_iter:

        grad_fk = f_prime(xk)
        x_next = g_prox(xk - step_size_x * grad_fk - step_size_x * L.T.dot(yk),
                        step_size_x * alpha, *g_prox_args)
        y_next = h_prox_conj(yk + step_size_y * L.dot(2 * x_next - xk),
                             step_size_y, *h_prox_args)

        incr = linalg.norm(x_next - xk) ** 2 + linalg.norm(y_next - yk) ** 2
        yk = y_next
        xk = x_next

        if verbose > 0:
            print("Iteration %s, increment: %s" % (it, incr))

        if callback is not None:
            callback(xk)

        if incr < tol:
            if verbose:
                print("Achieved relative tolerance at iteration %s" % it)
            success = True
            break

        it += 1

    if it >= max_iter:
        warnings.warn(
            "proximal_gradient did not reach the desired tolerance level", RuntimeWarning)

    return optimize.OptimizeResult(
        x=xk, success=success,
        nit=it)
