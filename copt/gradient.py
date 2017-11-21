import warnings
import numpy as np
from scipy import optimize, linalg, sparse
from tqdm import trange


def minimize_PGD(
        f_grad, x0, g_prox=None, tol=1e-6, max_iter=500, verbose=0,
        callback=None, backtracking=True, step_size=None,
        max_iter_backtracking=1000, backtracking_factor=0.6,
        ):
    """Proximal gradient descent.

    Solves problems of the form

            minimize_x f(x) + g(x)

    where we have access to the gradient of f and to the proximal operator of g.

    Arguments:
        f_grad: callable
             Returns the function value and gradient of the objective function.

        g : penalty term (proximal)

        x0 : array-like, optional
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
    x = x0
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')

    if g_prox is None:
        g_prox = lambda x, y: x
    else:
        raise ValueError

    if step_size is None:
        step_size = 1

    success = False
    certificate = np.NaN

    it = 1
    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..

    fk, grad_fk = f_grad(x)
    while it <= max_iter:
        if callback is not None:
            cb_args = {'x': x, 'grad': grad_fk, 'f': fk, 'gm': certificate}
            callback(cb_args)
        # .. compute gradient and step size
        current_step_size = step_size
        # TODO: could compute loss and grad in the same function call
        x_next = g_prox(x - current_step_size * grad_fk, current_step_size)
        incr = x_next - x
        if backtracking:
            for _ in range(max_iter_backtracking):
                f_next, grad_next = f_grad(x_next)
                if f_next <= fk + grad_fk.dot(incr) + incr.dot(incr) / (2.0 * current_step_size):
                    # .. step size found ..
                    break
                else:
                    # .. backtracking, reduce step size ..
                    current_step_size *= backtracking_factor
                    x_next = g_prox(x - current_step_size * grad_fk, current_step_size)
                    incr = x_next - x
            else:
                warnings.warn("Maxium number of line-search iterations reached")
        else:
            f_next, grad_next = f_grad(x_next)
        certificate = np.linalg.norm((x - x_next) / step_size)
        x[:] = x_next
        fk = f_next
        grad_fk = grad_next

        if verbose > 0:
            print("Iteration %s, step size: %s, certificate: %s" % (it, step_size, certificate))

        if certificate < tol:
            if verbose:
                print("Achieved relative tolerance at iteration %s" % it)
            success = True
            break

        if callback is not None:
            callback(locals())
        it += 1
    if it >= max_iter:
        warnings.warn(
            "proximal_gradient did not reach the desired tolerance level",
            RuntimeWarning)

    return optimize.OptimizeResult(
        x=x, success=success,
        certificate=certificate,
        nit=it)


def minimize_APGD(
        f_grad, x0, g_prox=None, tol=1e-6, max_iter=500, verbose=0,
        callback=None, backtracking=True,
        step_size=None, max_iter_backtracking=100, backtracking_factor=0.6,
        trace=False):
    """Accelerated proximal gradient descent.

    Solves problems of the form

            minimize_x f(x) + alpha g(x)

    where we have access to the gradient of f and to the proximal operator of g.

    Arguments:
        f_grad : loss function, differentiable

        g_prox : penalty, proximable

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
        Amir Beck and Marc Teboulle. "Gradient-based algorithms with applications to signal
        recovery." Convex optimization in signal processing and communications (2009)
    """
    x = x0
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')
    if g_prox is None:
        g_prox = lambda x, s: x

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
        x = g_prox(yk - current_step_size * grad_fk, current_step_size)
        if backtracking:
            for _ in range(max_iter_backtracking):
                incr = x - yk
                if f_grad(x)[0] <= f_grad(yk)[0] + grad_fk.dot(incr) + incr.dot(incr) / (2.0 * current_step_size):
                    # .. step size found ..
                    break
                else:
                    # .. backtracking, reduce step size ..
                    current_step_size *= backtracking_factor
                    x = g_prox(yk - current_step_size * grad_fk, current_step_size)
            else:
                warnings.warn("Maxium number of line-search iterations reached")
        t_next = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        yk = x + ((tk-1.) / t_next) * (x - xk_prev)
        certificate = np.linalg.norm((x - xk_prev) / step_size)
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
            callback(locals())
        it += 1
    if it >= max_iter:
        warnings.warn(
            "proximal_gradient did not reach the desired tolerance level",
            RuntimeWarning)

    return optimize.OptimizeResult(
        x=yk, success=success,
        certificate=certificate,
        nit=it)


def minimize_TOS(
        f_grad, g_prox=None, h_prox=None, x0=None, tol=1e-6, max_iter=1000,
        verbose=0, callback=None, backtracking=True, restart=True, step_size=None,
        max_iter_backtracking=100, backtracking_factor=0.5, trace=False, increase_rho=True):
    """Davis-Yin three operator splitting method.

    This algorithm can solve problems of the form

               minimize_x f(x) + g(x) + h(x)

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
    x0 = np.array(x0, copy=True)
    # y = np.array(y0, copy=True)
    success = False
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')

    if g_prox is None:
        g_prox = lambda x, s: x
    if h_prox is None:
        h_prox = lambda x, s: x

    if step_size is None:
        backtracking = True
        step_size = 1.

    y = x0 - h_prox(np.zeros(x0.size), step_size)
    LS_EPS = np.finfo(np.float).eps * 1e2
    rho = 1

    pbar = trange(max_iter)
    for it in pbar:
        x = h_prox(y, step_size)
        fk, grad_fk = f_grad(x)
        z = g_prox(
            x + rho * (x - y) - step_size * rho * grad_fk,
            rho * step_size)
        incr = z - x
        norm_incr = linalg.norm(incr)
        prox_grad_norm = norm_incr / (rho * step_size)
        if backtracking:
            # if restart and (rho > 10 or rho < 0.1):
            #     # lets do a restart
            #     y = z + rho * (y - z)
            #     step_size = step_size * rho
            #     rho = 1
            #     continue
            for it_ls in range(max_iter_backtracking):
                rhs = fk + grad_fk.dot(incr) \
                      + (norm_incr ** 2) / (2 * rho * step_size)
                ls_tol = f_grad(z, return_gradient=False) - rhs
                if ls_tol/fk <= LS_EPS:
                    # step size found
                    break
                else:
                    rho *= backtracking_factor
                    z = g_prox(
                        x + rho * (x - y) - step_size * rho * grad_fk,
                        rho * step_size)
                    incr = z - x
                    norm_incr = linalg.norm(incr)
            else:
                warnings.warn("Maximum number of line-search iterations reached")
            if it_ls == 0 and abs(ls_tol/fk) > LS_EPS:
                rho *= 1.07
        # if prox_grad_norm < 1e-12:
        #     backtracking = False
        pbar.set_description('Iteration %i' % it)
        pbar.set_postfix(tol=norm_incr / (rho * step_size), iter=it, rho=rho)

        if callback is not None:
            out = callback(locals())
            if out is False:
                break

        y += incr

        # if verbose > 0:
        #     # if it % 100 == 0:
        #     print("Iteration %s, prox-grad norm: %s, step size: %s, rho: %s" % (
        #             it, norm_incr / (rho * step_size), rho * step_size, rho))

        if prox_grad_norm < tol:
            success = True
            if verbose:
                print("Achieved relative tolerance at iteration %s" % it)
            break

        if it >= max_iter:
            warnings.warn(
                "three_split did not reach the desired tolerance level",
                RuntimeWarning)
        it += 1
    pbar.close()
    return optimize.OptimizeResult(
        x=z, success=success,
        jac=incr / (tol * step_size),  # prox-grad mapping
        nit=it,
        certificate=prox_grad_norm)
