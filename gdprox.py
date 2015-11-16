import numpy as np
from scipy import optimize
from scipy import linalg

__version__ = '0.1'


def fmin_cgprox(f, fprime, g_prox, x0, rtol=1e-6,
                maxiter=1000, verbose=0, default_step_size=1.):
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

    maxiter : int
        Maximum number of iterations.

    verbose : int
        Verbosity level, from 0 (no output) to 2 (output on each iteration)

    default_step_size : float
        Starting value for the line-search procedure.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.
    """
    xk = x0
    fk_old = np.inf

    fk, grad_fk = f(xk), fprime(xk)
    success = False
    for it in range(maxiter):
        # .. step 1 ..
        # Find suitable step size
        step_size = default_step_size  # initial guess
        grad_fk = fprime(xk)
        while True:  # adjust step size
            xk_grad = xk - step_size * grad_fk
            prx = g_prox(xk_grad, step_size)
            Gt = (xk - prx) / step_size
            lhand = f(xk - step_size * Gt)
            rhand = fk - step_size * grad_fk.dot(Gt) + \
                (0.5 * step_size) * Gt.dot(Gt)
            if lhand <= rhand:
                # step size found
                break
            else:
                # backtrack, reduce step size
                step_size *= .5

        xk -= step_size * Gt
        fk_old = fk
        fk, grad_fk = f(xk), fprime(xk)

        if verbose > 1:
            print("Iteration %s, Error: %s" % (it, linalg.norm(Gt)))

        if np.abs(fk_old - fk) / fk < rtol:
            if verbose:
                print("Achieved relative tolerance at iteration %s" % it)
                success = True
            break
    return optimize.OptimizationResult(
        x=xk, success=success, fun=fk, jac=grad_fk, nit=it)
