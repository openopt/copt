import warnings
import os
import numpy as np
from scipy import optimize, linalg, sparse
from tqdm import trange
DISABLE_TQDM = bool(os.environ.get('DISABLE_TQDM', False))


def minimize_PGD(
        f_grad, x0, g_prox=None, tol=1e-6, max_iter=500, verbose=0,
        callback=None, line_search=True, step_size=None,
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

        line_search : boolean
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
            callback(x)
        # .. compute gradient and step size
        # TODO: could compute loss and grad in the same function call
        x_next = g_prox(x - step_size * grad_fk, step_size)
        incr = x_next - x
        if line_search:
            step_size *= 1.1
            for _ in range(max_iter_backtracking):
                f_next, grad_next = f_grad(x_next)
                if f_next <= fk + grad_fk.dot(incr) + incr.dot(incr) / (2.0 * step_size):
                    # .. step size found ..
                    break
                else:
                    # .. backtracking, reduce step size ..
                    step_size *= backtracking_factor
                    x_next = g_prox(x - step_size * grad_fk, step_size)
                    incr = x_next - x
            else:
                warnings.warn("Maxium number of line-search iterations reached")
        else:
            f_next, grad_next = f_grad(x_next)
        certificate = np.linalg.norm((x - x_next) / step_size)
        x[:] = x_next
        fk = f_next
        grad_fk = grad_next

        pbar.set_description('Iteration %i' % it)
        pbar.set_postfix(tol=certificate, step_size=step_size, iter=it)

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
        nit=it)


def minimize_APGD(
        f_grad, x0, g_prox=None, tol=1e-6, max_iter=500, verbose=0,
        callback=None, line_search=True,
        step_size=None, max_iter_backtracking=100, backtracking_factor=0.6):
    """Accelerated proximal gradient descent.

    Solves problems of the form

            minimize_x f(x) + alpha g(x)

    where we have access to the gradient of f and to the proximal operator of g.

    Arguments:
        f_grad : loss function, differentiable

        g_prox : penalty, proximable

        g_prox : g_prox(x, alpha) returns the proximal operator of g at x
            with parameter alpha.

        x0 : array-like
            Initial guess

        line_search : boolean
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
        if line_search:
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
            callback(x)
        it += 1
    if it >= max_iter:
        warnings.warn(
            "minimize_APGD did not reach the desired tolerance level",
            RuntimeWarning)

    return optimize.OptimizeResult(
        x=yk, success=success,
        certificate=certificate,
        nit=it)


def minimize_TOS(
        f_grad, x0, g_prox=None, h_prox=None, tol=1e-6, max_iter=1000,
        verbose=0, callback=None, line_search=True, step_size=None,
        max_iter_backtracking=100, backtracking_factor=0.7, h_Lipschitz=None):
    """Davis-Yin three operator splitting method.

    This algorithm can solve problems of the form

               minimize_x f(x) + g(x) + h(x)

    where f is a smooth function and g is a (possibly non-smooth)
    function for which the proximal operator is known.

    Parameters
    ----------
    f_grad: callable
         Returns the function value and gradient of the objective function.
         With return_gradient=False, returns only the function value.

    g_prox : callable or None
        g_prox(x, alpha, *args) returns the proximal operator of g at xa
        with parameter alpha. Extra arguments can be passed by g_prox_args.

    y0 : array-like
        Initial guess

    line_search : boolean
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
    * Davis, Damek, and Wotao Yin. `"A three-operator splitting scheme and its optimization
      applications." <https://doi.org/10.1007/s11228-017-0421-z>`_ Set-Valued and Variational Analysis, 2017.

    * Pedregosa, Fabian, and Gauthier Gidel. `"Adaptive Three Operator Splitting."
      <https://arxiv.org/abs/1804.02339>`_ Proceedings of the 35th International Conference
      on Machine Learning, 2018.
    """
    success = False
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')

    if g_prox is None:
        g_prox = lambda x, s: x
    if h_prox is None:
        h_prox = lambda x, s: x

    if step_size is None:
        line_search = True
        step_size = 1.

    z = h_prox(x0, step_size)
    LS_EPS = np.finfo(np.float).eps

    fk, grad_fk = f_grad(z)
    x = g_prox(z - step_size * grad_fk, step_size)
    u = np.zeros_like(x)
    ls_tol_old = None

    pbar = trange(max_iter, disable=DISABLE_TQDM)
    pbar.set_description('TOS')
    for it in pbar:

        if line_search:
            for it_ls in range(max_iter_backtracking):
                fk, grad_fk = f_grad(z)
                x = g_prox(z - step_size * (u + grad_fk), step_size)
                incr = x - z
                norm_incr = np.linalg.norm(incr)
                rhs = fk + grad_fk.dot(incr) + (norm_incr ** 2) / (2 * step_size)
                ls_tol = f_grad(x, return_gradient=False) - rhs
                if ls_tol <= LS_EPS:
                    # step size found
                    if ls_tol > 0:
                        ls_tol = 0.
                    break
                else:
                    step_size *= backtracking_factor
        else:
            fk, grad_fk = f_grad(z)
            x = g_prox(z - step_size * (u + grad_fk), step_size)
            incr = x - z
            norm_incr = np.linalg.norm(incr)

        z = h_prox(x + step_size * u, step_size)
        u += (x - z) / step_size
        certificate = norm_incr / step_size

        if line_search and h_Lipschitz is not None:
            if h_Lipschitz == 0:
                step_size = step_size * 1.02
            else:
                quot = h_Lipschitz ** 2
                tmp = np.sqrt(
                    step_size ** 2 + (2 * step_size / quot) * (-ls_tol))
                step_size = min(tmp, step_size * 1.02)
        if it % 10 == 0:
            pbar.set_postfix(tol=certificate, iter=it, step_size=step_size)

        if callback is not None:
            if callback(x) is False:
                break

        if it > 0 and certificate < tol:
            success = True
            if verbose:
                pbar.write("Achieved relative tolerance at iteration %s" % it)
            break

        if it >= max_iter:
            pbar.write(
                "warning: three_split did not reach the desired tolerance level")
    pbar.close()
    return optimize.OptimizeResult(
        x=x, success=success, nit=it,
        certificate=certificate)


def minimize_PDHG(
        f_grad, x0, g_prox=None, h_prox=None, L=None, tol=1e-12,
        max_iter=1000, callback=None, step_size=1., step_size2=None,
        line_search=True, max_iter_ls=20):
    """Primal-dual hybrid gradient splitting method.

    This method for optimization problems of the form

            minimize_x f(x) + alpha * g(x) + beta * h(L x)

    where f is a smooth function and g is a (possibly non-smooth)
    function for which the proximal operator is known.

    Parameters
    ----------
    f_grad: callable
         Returns the function value and gradient of the objective function.
         With return_gradient=False, returns only the function value.

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
    Condat, Laurent. "A primal-dual splitting method for convex optimization
    involving Lipschitzian, proximable and linear composite terms." Journal of
    Optimization Theory and Applications (2013).

    Chambolle, Antonin, and Thomas Pock. "On the ergodic convergence rates of a
    first-order primal-dual algorithm." Mathematical Programming (2015)
    """
    x = np.array(x0, copy=True)
    n_features = x.size
    if L is None:
        L = sparse.eye(n_features, n_features, format='csr')
        def Ldot(x): return x
        Ltdot = Ldot
    y = L.dot(x)
    success = False
    if not max_iter_ls > 0:
        raise ValueError('Line search iterations need to be greater than 0')

    if g_prox is None:
        def g_prox(x, step_size): return x
    if h_prox is None:
        def h_prox(x, step_size): return x

    # conjugate of h_prox
    def h_prox_conj(x, ss):
        return x - ss * h_prox(x / ss, 1. / ss)
    # .. main iteration ..
    theta = 1.
    delta = 0.99
    sigma = step_size
    tau = step_size2
    if tau is None:
        tau = 0.5 * sigma
    ss_ratio = sigma / tau

    pbar = trange(max_iter, disable=DISABLE_TQDM)
    fk, grad_fk = f_grad(x)
    norm_incr = np.infty
    x_next = x.copy()

    for it in pbar:
        y_next = h_prox_conj(y + tau * Ldot(x), tau)
        if line_search:
            tau_next = tau * np.sqrt(1 + theta)
            while True:
                theta = tau_next / tau
                sigma = ss_ratio * tau_next
                y_bar = y_next + theta * (y_next - y)
                x_next = g_prox(x - sigma * (Ltdot(y_bar) + grad_fk), sigma)
                incr_x = np.linalg.norm(Ltdot(x_next) - Ltdot(x))
                if incr_x <= 1e-10:
                    break

                f_next, f_grad_next = f_grad(x_next)
                tmp = (sigma * tau_next) * (incr_x ** 2)
                tmp += 2 * sigma * (f_next - fk - grad_fk.dot(x_next - x))
                if tmp <= delta * (incr_x ** 2) + np.finfo(np.float).eps:
                    tau = tau_next
                    break
                else:
                    tau_next *= 0.5
        else:
            y_bar = 2 * y_next - y
            x_next = g_prox(x - sigma * (Ltdot(y_bar) + grad_fk), sigma)
            f_next, f_grad_next = f_grad(x_next)

        if it % 100 == 0:
            norm_incr = linalg.norm(x_next - x) + linalg.norm(y_next - y)
            pbar.set_description('PDHG iter %i' % it)
            pbar.set_postfix(tol=norm_incr, iter=it, step_size=sigma, step_size2=tau, quot=sigma * tau)

        x[:] = x_next[:]
        y[:] = y_next[:]
        fk, grad_fk = f_next, f_grad_next

        if norm_incr < tol:
            success = True
            break

        if callback is not None:
            if callback(x_next) is False:
                break

    if it >= max_iter:
        warnings.warn(
            "proximal_gradient did not reach the desired tolerance level", RuntimeWarning)

    pbar.close()
    return optimize.OptimizeResult(
        x=y, success=success, nit=it, certificate=norm_incr)