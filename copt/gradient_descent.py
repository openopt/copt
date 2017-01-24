import warnings
from typing import Callable
import numpy as np
from scipy import optimize
from scipy import linalg


def fmin_PGD(fun: Callable, fun_deriv: Callable, g_prox, x0: np.ndarray, alpha=1.0, tol=1e-6, max_iter=1000,
             verbose=0, g_prox_args=(), callback=None, backtracking: bool=True,
             step_size=None, max_iter_backtracking=100, backtracking_factor=0.4
             ) -> optimize.OptimizeResult:
    """Proximal gradient descent.

    Solves problems of the form

            minimize_x f(x) + alpha g(x)


    where we have access to the gradient of f and to the proximal operator of g.

    Arguments:
        fun : f(x) returns the value of f at x.

        fun_deriv : f_prime(x) returns the gradient of f.

        g_prox : g_prox(x, alpha) returns the proximal operator of g at x
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

    Returns:
        res : The optimization result represented as a
            ``scipy.optimize.OptimizeResult`` object. Important attributes are:
            ``x`` the solution array, ``success`` a Boolean flag indicating if
            the optimizer exited successfully and ``message`` which describes
            the cause of the termination. See `scipy.optimize.OptimizeResult`
            for a description of other attributes.

    References:
        Beck, Amir, and Marc Teboulle. "Gradient-based algorithms with applications to signal
        recovery." Convex optimization in signal processing and communications (2009)
    """
    xk = np.array(x0, copy=True)
    success = False
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')
    if g_prox is None:
        g_prox = lambda x, y: y

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
        x_next = g_prox(current_step_size * alpha, xk - current_step_size * grad_fk, *g_prox_args)
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
                    x_next = g_prox(current_step_size * alpha, xk - current_step_size * grad_fk, *g_prox_args)
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


def fmin_DavisYin(
        fun, fun_deriv, g_prox, h_prox, y0, alpha=1.0, beta=1.0, tol=1e-6, max_iter=1000,
        g_prox_args=(), h_prox_args=(),
        verbose=0, callback=None, backtracking=True, step_size=None, max_iter_backtracking=100,
        backtracking_factor=0.4):
    """Davis-Yin three operator splitting method.

    This algorithm can solve problems of the form

               minimize_x f(x) + alpha * g(x) + beta * h(x)

    where f is a smooth function and g is a (possibly non-smooth)
    function for which the proximal operator is known.

    Parameters
    ----------
    fun : callable
        f(x) returns the value of f at x.

    fun_deriv : callable or None
        f_prime(x) returns the gradient of f.

    g_prox : callable or None
        g_prox(x, alpha, *args) returns the proximal operator of g at xa
        with parameter alpha. Extra arguments can be passed by g_prox_args.

    y0 : array-like
        Initial guess

    backtracking : boolean
        Whether to perform backtracking (i.e. line-search) to estimate
        the step size.

    max_iter : int
        Maximum number of iterations.

    verbose : int
        Verbosity level, from 0 (no output) to 2 (output on each iteration)

    step_size : float
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
    Davis, Damek, and Wotao Yin. "A three-operator splitting scheme and its optimization applications."
    arXiv preprint arXiv:1504.01032 (2015) https://arxiv.org/abs/1504.01032

    Pedregosa, Fabian. "On the convergence rate of the three operator splitting scheme." arXiv preprint
    arXiv:1610.07830 (2016) https://arxiv.org/abs/1610.07830
    """
    y = np.array(y0, copy=True)
    success = False
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')

    if g_prox is None:
        def g_prox(step_size, x, *args): return x
    if h_prox is None:
        def h_prox(step_size, x, *args): return x

    if step_size is None:
        # sample to estimate Lipschitz constant
        x0 = np.zeros(y.size)
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
    current_step_size = step_size
    while it <= max_iter:
        x = g_prox(current_step_size * alpha, y, *g_prox_args)
        grad_fk = fun_deriv(x)
        z = h_prox(current_step_size * beta, 2 * x - y - current_step_size * grad_fk, *h_prox_args)
        incr = z - x
        norm_incr = linalg.norm(incr / current_step_size)
        if backtracking:
            for _ in range(max_iter_backtracking):
                expected_descent = fun(x) + grad_fk.dot(incr) + 0.5 * current_step_size * (norm_incr ** 2)
                if fun(z) <= expected_descent * (1 + x.size * np.finfo(np.float64).eps):
                    # step size found
                    break
                else:
                    current_step_size *= backtracking_factor
                    y = x + backtracking_factor * (y - x)
                    grad_fk = fun_deriv(x)
                    z = h_prox(current_step_size * beta, 2 * x - y - current_step_size * grad_fk, *h_prox_args)
                    incr = z - x
                    norm_incr = linalg.norm(incr / current_step_size)
            else:
                warnings.warn("Maximum number of line-search iterations reached")

        y += incr

        if verbose > 0:
            print("Iteration %s, prox-grad norm: %s, step size: %s" % (
                it, norm_incr / current_step_size, current_step_size))

        if norm_incr < tol * current_step_size:
            success = True
            if verbose:
                print("Achieved relative tolerance at iteration %s" % it)
            break

        if callback is not None:
            callback(x)
        if it >= max_iter:
            warnings.warn(
                "three_split did not reach the desired tolerance level",
                RuntimeWarning)
        it += 1

    x_sol = g_prox(current_step_size * alpha, y, *g_prox_args)
    return optimize.OptimizeResult(
        x=x_sol, success=success,
        jac=incr / current_step_size,  # prox-grad mapping
        nit=it)
