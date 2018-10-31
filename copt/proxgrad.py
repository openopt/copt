import warnings
import numpy as np
from scipy import optimize
from tqdm import trange


def minimize_PGD(
        f_grad, x0, prox=None, tol=1e-6, max_iter=500, verbose=0,
        callback=None, backtracking=True, step_size=None,
        max_iter_backtracking=1000, backtracking_factor=0.6,
        ):
    """Proximal gradient descent.

    Solves problems of the form

            minimize_x f(x) + g(x)

    where we have access to the gradient of f and the proximal operator of g.

    Parameters
    ----------
    f_grad: callable
         Value and gradient of f: ``f_grad(x) -> float, array-like``.

    x0 : array-like of size n_features
        Initial guess of solution.

    prox : callable, optional.
        Proximal operator g.

    backtracking : boolean, optional
        Whether to perform backtracking (i.e. line-search) or not.

    max_iter : int, optional.
        Maximum number of iterations.

    verbose : int, optional.
        Verbosity level, from 0 (no output) to 2 (output on each iteration)

    step_size : float
        Starting value for the line-search procedure. XXX

    callback : callable
        callback function (optional). Takes a single argument (x) with the
        current coefficients in the algorithm. The algorithm will exit if
        callback returns False.

    Returns
    -------
    res : The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.

    References
    ----------
    Beck, Amir, and Marc Teboulle. "Gradient-based algorithms with applications
    to signal recovery." Convex optimization in signal processing and
    communications (2009)

    Examples
    --------
      * :ref:`sphx_glr_auto_examples_plot_group_lasso.py`

    """
    x = x0
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')

    if prox is None:
        def prox(x, step_size):
            return x

    if step_size is None:
        step_size = 1

    success = False
    certificate = np.NaN

    it = 1
    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..

    fk, grad_fk = f_grad(x)
    pbar = trange(max_iter, disable=(verbose == 0))
    for it in pbar:
        if callback is not None:
            if callback(x) is False:
                break
        # .. compute gradient and step size
        x_next = prox(x - step_size * grad_fk, step_size)
        incr = x_next - x
        if backtracking:
            step_size *= 1.1
            for _ in range(max_iter_backtracking):
                f_next, grad_next = f_grad(x_next)
                if f_next <= fk + grad_fk.dot(incr) + incr.dot(incr) / (2.0 * step_size):
                    # .. step size found ..
                    break
                else:
                    # .. backtracking, reduce step size ..
                    step_size *= backtracking_factor
                    x_next = prox(x - step_size * grad_fk, step_size)
                    incr = x_next - x
            else:
                warnings.warn("Maxium number of line-search iterations reached")
        else:
            f_next, grad_next = f_grad(x_next)
        certificate = np.linalg.norm((x - x_next) / step_size)
        x[:] = x_next
        fk = f_next
        grad_fk = grad_next

        pbar.set_description('PGD')
        if backtracking:
            pbar.set_postfix(tol=certificate, step_size=step_size, iter=it)
        else:
            pbar.set_postfix(tol=certificate, iter=it)

        if certificate < tol:
            if verbose:
                pbar.write("Achieved relative tolerance at iteration %s" % it)
            success = True
            break
    else:
        warnings.warn(
            "minimize_PGD did not reach the desired tolerance level",
            RuntimeWarning)
    pbar.close()
    return optimize.OptimizeResult(
        x=x, success=success,
        certificate=certificate,
        nit=it, step_size=step_size)


def minimize_APGD(
        f_grad, x0, prox=None, tol=1e-6, max_iter=500, verbose=0,
        callback=None, backtracking=True,
        step_size=None, max_iter_backtracking=100, backtracking_factor=0.6):
    """Accelerated proximal gradient descent.

    Solves problems of the form

            minimize_x f(x) + g(x)

    where we have access to the gradient of f and the proximal operator of g.

    Parameters
    ----------
    f_grad : loss function, differentiable

    prox : prox(x, alpha) returns the proximal operator of g at x
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
    res : The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.

    References
    ----------
    Amir Beck and Marc Teboulle. "Gradient-based algorithms with applications
    to signal recovery." Convex optimization in signal processing and
    communications (2009)

    """
    x = x0
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')

    if prox is None:
        def prox(x, step_size):
            return x

    if step_size is None:
        step_size = 1

    success = False
    certificate = np.inf

    it = 1
    tk = 1
    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..
    yk = x.copy()
    xk_prev = x.copy()
    while it <= max_iter:
        # .. compute gradient and step size
        current_step_size = step_size
        grad_fk = f_grad(yk)[1]
        x = prox(yk - current_step_size * grad_fk, current_step_size)
        if backtracking:
            for _ in range(max_iter_backtracking):
                incr = x - yk
                if f_grad(x)[0] <= f_grad(yk)[0] + grad_fk.dot(incr) + incr.dot(incr) / (2.0 * current_step_size):
                    # .. step size found ..
                    break
                else:
                    # .. backtracking, reduce step size ..
                    current_step_size *= backtracking_factor
                    x = prox(yk - current_step_size * grad_fk, current_step_size)
            else:
                warnings.warn("Maxium number of line-search iterations reached")
        t_next = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        yk = x + ((tk-1.) / t_next) * (x - xk_prev)

        x_prox = prox(x - current_step_size * f_grad(x)[1], current_step_size)
        certificate = np.linalg.norm((x - x_prox) / step_size)
        tk = t_next
        xk_prev = x.copy()

        if verbose > 0:
            print("Iteration %s, certificate: %s, step size: %s" % (it, certificate, step_size))

        if False and certificate < tol:
            if verbose:
                print("Achieved relative tolerance at iteration %s" % it)
            success = True
            break

        if callback is not None:
            callback(x)
        it += 1
    if it >= max_iter:
        warnings.warn(
            "minimize_APGD did not reach the desired tolerance level",
            RuntimeWarning)

    return optimize.OptimizeResult(
        x=yk, success=success,
        certificate=certificate,
        nit=it, step_size=step_size)
