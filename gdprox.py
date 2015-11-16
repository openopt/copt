import numpy as np

__version__ = '0.1'


def fmin_cgprox(f, fprime, g_prox, x0, rtol=1e-6,
                maxiter=10000, verbose=False, default_step_size=1.):
    """
    solve optimization problems of the form

        minimize_x f(x) + g(x)

    Parameters
    ----------
    f : callable
    g_prox : callable of the form g_prox(x, alpha)
    """

    xk = x0
    fk_old = np.inf

    fk = f(xk)
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
        fk = f(xk)

        if np.abs(fk_old - fk) / fk < rtol:
            if verbose:
                print("Achieved relative tolerance at iteration %s" % it)
            break
    return xk

