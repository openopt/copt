import warnings
import numpy as np
from scipy import optimize, linalg, sparse
from datetime import datetime

# .. local imports ..
from .utils import ZeroLoss


def minimize_PGD(
        f, g=None, x0=None, tol=1e-6, max_iter=500, verbose=0,
        callback=None, backtracking: bool=True, step_size=None,
        max_iter_backtracking=100, backtracking_factor=0.6, trace=False
        ) -> optimize.OptimizeResult:
    """Proximal gradient descent.

    Solves problems of the form

            minimize_x f(x) + g(x)

    where we have access to the gradient of f and to the proximal operator of g.

    Arguments:
        f : loss function (smooth)

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
    if x0 is None:
        xk = np.zeros(f.n_features)
    else:
        xk = np.array(x0, copy=True)
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')
    if g is None:
        g = ZeroLoss()

    if step_size is None:
        # sample to estimate Lipschitz constant
        step_size_n_sample = 5
        L = []
        for _ in range(step_size_n_sample):
            x_tmp = np.random.randn(f.n_features)
            x_tmp /= linalg.norm(x_tmp)
            L.append(linalg.norm(f(xk) - f(x_tmp)))
        # give it a generous upper bound
        step_size = 2. / np.mean(L)

    success = False
    trace_func = []
    trace_time = []
    trace_x = []
    start_time = datetime.now()

    it = 1
    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..

    if trace:
        trace_x.append(xk.copy())
        trace_func.append(f(xk) + g(xk))
        trace_time.append((datetime.now() - start_time).total_seconds())

    while it <= max_iter:
        # .. compute gradient and step size
        current_step_size = step_size
        grad_fk = f.gradient(xk)
        # TODO: could compute loss and grad in the same function call
        x_next = g.prox(xk - current_step_size * grad_fk, current_step_size)
        incr = x_next - xk
        if backtracking:
            fk = f(xk)
            f_next = f(x_next)
            for _ in range(max_iter_backtracking):
                if f_next <= fk + grad_fk.dot(incr) + incr.dot(incr) / (2.0 * current_step_size):
                    # .. step size found ..
                    break
                else:
                    # .. backtracking, reduce step size ..
                    current_step_size *= backtracking_factor
                    x_next = g.prox(xk - current_step_size * grad_fk, current_step_size)
                    incr = x_next - xk
                    f_next = f(x_next)
            else:
                warnings.warn("Maxium number of line-search iterations reached")
        certificate = np.linalg.norm((xk - x_next) / step_size)
        xk[:] = x_next

        if trace:
            trace_x.append(xk.copy())
            trace_func.append(f(xk) + g(xk))
            trace_time.append((datetime.now() - start_time).total_seconds())

        if verbose > 0:
            print("Iteration %s, step size: %s, certificate: %s" % (it, step_size, certificate))

        if certificate < tol:
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
        certificate=certificate,
        nit=it, trace_x=np.array(trace_x), trace_func=np.array(trace_func),
        trace_time=trace_time)


def minimize_APGD(
        f, g=None, x0=None, tol=1e-6, max_iter=500, verbose=0,
        callback=None, backtracking: bool=True,
        step_size=None, max_iter_backtracking=100, backtracking_factor=0.6,
        trace=False) -> optimize.OptimizeResult:
    """Accelerated proximal gradient descent.

    Solves problems of the form

            minimize_x f(x) + alpha g(x)

    where we have access to the gradient of f and to the proximal operator of g.

    Arguments:
        f : loss function, differentiable

        g : penalty, proximable

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
    if x0 is None:
        xk = np.zeros(f.n_features)
    else:
        xk = np.array(x0, copy=True)
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')
    if g is None:
        g = ZeroLoss()

    if step_size is None:
        # sample to estimate Lipschitz constant
        step_size_n_sample = 5
        L = []
        for _ in range(step_size_n_sample):
            x_tmp = np.random.randn(f.n_features)
            x_tmp /= linalg.norm(x_tmp)
            L.append(linalg.norm(f(xk) - f(x_tmp)))
        # give it a generous upper bound
        step_size = 2. / np.mean(L)

    success = False
    trace_func = []
    trace_time = []
    trace_certificate = []
    start_time = datetime.now()
    certificate = np.inf

    it = 1
    tk = 1
    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..
    yk = xk.copy()
    xk_prev = xk.copy()
    while it <= max_iter:
        # .. compute gradient and step size
        current_step_size = step_size
        grad_fk = f.gradient(yk)
        xk = g.prox(yk - current_step_size * grad_fk, current_step_size)
        if backtracking:
            for _ in range(max_iter_backtracking):
                incr = xk - yk
                if f(xk) <= f(yk) + grad_fk.dot(incr) + incr.dot(incr) / (2.0 * current_step_size):
                    # .. step size found ..
                    break
                else:
                    # .. backtracking, reduce step size ..
                    current_step_size *= backtracking_factor
                    xk = g.prox(yk - current_step_size * grad_fk, current_step_size)
            else:
                warnings.warn("Maxium number of line-search iterations reached")
        t_next = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        yk = xk + ((tk-1.) / t_next) * (xk - xk_prev)
        certificate = np.linalg.norm((xk - xk_prev) / step_size)
        tk = t_next
        xk_prev = xk.copy()

        if trace:
            trace_certificate.append(certificate)
            trace_func.append(f(yk) + g(yk))
            trace_time.append((datetime.now() - start_time).total_seconds())

        if verbose > 0:
            print("Iteration %s, certificate: %s, step size: %s" % (it, certificate, step_size))

        if False and certificate < tol:
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
        x=yk, success=success,
        certificate=certificate,
        trace_certificate=trace_certificate,
        nit=it, trace_func=np.array(trace_func),
        trace_time=trace_time)

def minimize_DavisYin(
        f, g=None, h=None, x0=None, tol=1e-6, max_iter=1000,
        verbose=0, callback=None, backtracking=True, restart=True, step_size=None,
        max_iter_backtracking=100, backtracking_factor=0.4, trace=False):
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
    if x0 is None:
        x0 = np.zeros(f.n_features)
    else:
        x0 = np.array(x0, copy=True)
    # y = np.array(y0, copy=True)
    success = False
    if not max_iter_backtracking > 0:
        raise ValueError('Line search iterations need to be greater than 0')

    if g is None:
        g = ZeroLoss()
    if h is None:
        h = ZeroLoss()

    if step_size is None:
        if backtracking:
            L = f.lipschitz_constant('full')
            step_size = 1 / L
        else:
            L = f.lipschitz_constant('full')
            step_size = 1 / L

    y = x0 - g.prox(np.zeros(x0.size), step_size)
    trace_func = []
    trace_time = []
    start_time = datetime.now()

    it = 1
    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..
    rho = 1
    if callback is not None:
        callback(x0)
    while it <= max_iter:
        z = g.prox(y, step_size)
        grad_fk = f.gradient(z)
        x = h.prox(z + rho * (z - y) - step_size * rho * grad_fk, rho * step_size)
        incr = x - z
        norm_delta = linalg.norm(incr)
        prox_grad_norm = norm_delta / (rho * step_size)
        if backtracking:
            if restart and (rho > 100 or rho < 0.01):
                rho = 1
                y = z + rho * (y - z)
                step_size *= rho
                rho = 1
                x = h.prox(z + rho * (z - y) - step_size * rho * grad_fk, rho * step_size)
                incr = x - z
            fz = f(z)
            it_ls = 0
            while it_ls < max_iter_backtracking:
                rhs = fz + grad_fk.dot(incr) + 0.5 * (norm_delta ** 2) / (rho * step_size)
                if f(x) <= rhs:
                    # step size found
                    break
                else:
                    rho *= backtracking_factor
                    x = h.prox(z + rho * (z - y) - step_size * rho * grad_fk, rho * step_size)
                    incr = x - z
                    norm_delta = linalg.norm(incr)
                it_ls += 1
            else:
                warnings.warn("Maximum number of line-search iterations reached")
            if it_ls == 0:
                # .. so that it doubles every ~20 successful iterations ..
                rho *= 1.03

        if trace:
            trace_time.append((datetime.now() - start_time).total_seconds())
            if backtracking:
                trace_func.append(f(x) + g(x) + h(x))
            else:
                trace_func.append(f(x) + g(x) + h(x))

        y += incr

        if verbose > 0:
            # if it % 100 == 0:
            print("Iteration %s, prox-grad norm: %s, step size: %s, rho: %s" % (
                    it, norm_delta / (rho * step_size), rho * step_size, rho))

        if prox_grad_norm < tol:
            success = True
            if verbose:
                print("Achieved relative tolerance at iteration %s" % it)
            break

        if callback is not None:
            callback(0.5 * (x + z))
        if it >= max_iter:
            warnings.warn(
                "three_split did not reach the desired tolerance level",
                RuntimeWarning)
        it += 1

    if len(trace_time) > 0:
        trace_time = np.array(trace_time) - trace_time[0]
    return optimize.OptimizeResult(
        x=x, success=success,
        jac=incr / (tol * step_size),  # prox-grad mapping
        nit=it, trace_func=np.array(trace_func),
        trace_time=trace_time, certificate=prox_grad_norm)
