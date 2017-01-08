import warnings
import numpy as np
from scipy import optimize
from scipy import linalg


def fmin_ProxGrad(fun, fun_deriv, g_prox, x0, alpha=1.0, tol=1e-6, max_iter=1000,
                  verbose=0, g_prox_args=(), callback=None, backtracking=True,
                  step_size=None, max_iter_backtracking=100, backtracking_factor=0.4):
    """Proximal gradient descent.

    Solves problems of the form

                       minimize_x f(x) + alpha * g(x)

    where we have access to the gradient of f and to the proximal operator of g.

    Parameters
    ----------
    fun : callable
        f(x) returns the value of f at x.

    fun_deriv : callable
        f_prime(x) returns the gradient of f.

    g_prox : callable or None
        g_prox(x, alpha) returns the proximal operator of g at x
        with parameter alpha.

    x0 : array-like
        Initial guess

    backtracking : boolean
        Whether to perform backtracking (i.e. line-search) or not.

    max_iter : int
        Maximum number of iterations.

    verbose : int
        Verbosity level, from 0 (no output) to 2 (output on each iteration)

    step_size : float
        Starting value for the line-search procedure. XXX

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
    Beck, Amir, and Marc Teboulle. "Gradient-based algorithms with applications to signal
    recovery." Convex optimization in signal processing and communications (2009)
    """
    xk = np.array(x0, copy=True)
    success = False
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')
    if g_prox is None:
        g_prox = lambda x, y: x

    if step_size is None:
        # sample to estimate Lipschitz constant
        step_size_n_sample = 5
        L = []
        for _ in range(step_size_n_sample):
            x_tmp = np.random.randn(x0.size)
            x_tmp /= linalg.norm(x_tmp)
            L.append(linalg.norm(fun_deriv(x0) - fun_deriv(x_tmp)))
        # give it a generous upper bound
        step_size = 10. / np.mean(L)

    it = 1
    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..
    while it <= max_iter:
        # .. compute gradient and step size
        current_step_size = step_size
        grad_fk = fun_deriv(xk)
        x_next = g_prox(xk - current_step_size * grad_fk, current_step_size * alpha, *g_prox_args)
        incr = x_next - xk
        if backtracking:
            fk = fun(xk)
            f_next = fun(x_next)
            for _ in range(max_iter_backtracking):
                if f_next <= fk + grad_fk.dot(incr) + incr.dot(incr) / (2.0 * current_step_size):
                    # .. step size found ..
                    break
                else:
                    # .. backtracking, reduce step size ..
                    current_step_size *= backtracking_factor
                    x_next = g_prox(xk - current_step_size * grad_fk, current_step_size * alpha, *g_prox_args)
                    incr = x_next - xk
                    f_next = fun(x_next)
            else:
                warnings.warn("Maxium number of line-search iterations reached")
        xk = x_next

        norm_increment = linalg.norm(incr, np.inf) / current_step_size
        if verbose > 0:
            print("Iteration %s, prox-grad norm: %s" % (it, norm_increment))

        if norm_increment < tol:
            if verbose:
                print("Achieved relative tolerance at iteration %s" % it)
            success = True
            break

        if callback is not None:
            callback(xk)
        it += 1
    if it >= max_iter:
        warnings.warn(
            "proximal_gradient did not reach the desired tolerance level",
            RuntimeWarning)

    return optimize.OptimizeResult(
        x=xk, success=success,
        jac=incr / step_size,  # prox-grad mapping
        nit=it)
