from datetime import datetime
import numpy as np
from scipy import sparse, optimize
from numba import njit
from copt import utils
from tqdm import tqdm

# def njit(*args, **kwargs):
#     return lambda x: x

@njit
def f_squared(p, y):
    # squared loss
    return 0.5 * ((y - p) ** 2)


@njit
def deriv_squared(p, y):
    # derivative of squared loss
    return - (y - p)


@njit
def f_logistic(p, y):
    # logistic loss
    # same as in lightning
    p *= y
    if p > 0:
        return np.log(1 + np.exp(-p))
    else:
        return -p + np.log(1 + np.exp(p))


@njit
def deriv_logistic(p, y):
    # derivative of logistic loss
    # same as in lightning (with minus sign)
    p *= y
    if p > 0:
        phi = 1. / (1 + np.exp(-p))
    else:
        exp_t = np.exp(p)
        phi = exp_t / (1. + exp_t)
    return (phi - 1) * y



def minimize_SAGALP_L1(
        f_deriv, A, b, x0, alpha=0, beta=0, step_size=None,
        max_iter=500, tol=1e-6, callback=None):
    """Stochastic average gradient augmented (SAGA) algorithm.

    The SAGA algorithm can solve optimization problems of the form

        argmin_{x \in R^p} \sum_{i}^n_samples f(A_i^T x, b_i) + alpha * ||x||_2^2 +
                                            + beta * ||x||_1

    Parameters
    ----------
    f, g
        loss functions. g can be none

    x0: np.ndarray or None, optional
        Starting point for optimization.

    step_size: float or None, optional
        Step size for the optimization. If None is given, this will be
        estimated from the function f.

    max_iter: int
        Maximum number of passes through the data in the optimization.

    tol: float
        Tolerance criterion. The algorithm will stop whenever the norm of the
        gradient mapping (generalization of the gradient for nonsmooth optimization)
        is below tol.

    verbose: bool
        Verbosity level. True might print some messages.

    trace: bool
        Whether to trace convergence of the function, useful for plotting and/or
        debugging. If ye, the result will have extra members trace_func,
        trace_time.

    Returns
    -------
    opt: OptimizeResult
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.

    References
    ----------
    This variant of the SAGA algorithm is described in

        Fabian Pedregosa, Remi Leblond, and Simon Lacoste-Julien. "Breaking the Nonsmooth
        Barrier: A Scalable Parallel Method for Composite Optimization." Advances in
        Neural Information Processing Systems (NIPS) 2017.
    """
    # convert any input to CSR sparse matrix representation. In the future we might want to
    # implement also a version for dense data (numpy arrays) to better exploit data locality
    x = np.ascontiguousarray(x0).copy()
    n_samples, n_features = A.shape

    if step_size is None:
        # then need to use line search
        raise ValueError

    # .. estimate diagonal elements of the reweighting matrix (D) ..
    print('Computing D matrix')
    tmp = A.copy()
    tmp.data[:] = 1.
    d = np.array(tmp.sum(0), dtype=np.float).ravel()
    idx = (d != 0)
    d[idx] = n_samples / d[idx]
    d[~idx] = 0.
    print('Done')

    if beta > 0:
        @njit
        def prox(x, ss):
            return np.fmax(x - beta * ss, 0) - np.fmax(- x - beta * ss, 0)
    elif beta == 0:
        @njit
        def prox(x, ss):
            return x
    else:
        raise ValueError

    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    n_samples, n_features = A.shape

    @njit(nogil=True)
    def _saga_epoch(
            x, idx, memory_gradient, gradient_average, step_size):
        # .. inner iteration of the SAGA algorithm..
        for i in idx:
            p = 0.
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                p += x[j_idx] * A_data[j]

            grad_i = f_deriv(p, b[i])

            # .. update coefficients ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                delta = (grad_i - memory_gradient[i]) * A_data[j]
                incr = delta + d[j_idx] * (gradient_average[j_idx] + alpha * x[j_idx])
                x[j_idx] = prox(x[j_idx] - step_size * incr, step_size * d[j_idx])

            # .. update memory terms ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                gradient_average[j_idx] += (grad_i - memory_gradient[i]) * A_data[j] / n_samples
            memory_gradient[i] = grad_i

    # .. initialize memory terms ..
    memory_gradient = np.zeros(n_samples)
    gradient_average = np.zeros(n_features)
    idx = np.arange(n_samples)
    success = False
    if callback is not None:
        callback(x)
    for nit in range(max_iter):
        x_old = x.copy()
        np.random.shuffle(idx)
        _saga_epoch(
                x, idx, memory_gradient, gradient_average, step_size)
        if callback is not None:
            callback(x)

        if np.abs(x - x_old).sum() < tol:
            success = True
            break
        print(nit, np.linalg.norm(x - x_old))
    message = ''
    return optimize.OptimizeResult(
        x=x, success=success, nit=nit,
        message=message)




def minimize_BCD(
        f, g=None, x0=None, step_size=None, max_iter=300, trace=False, verbose=False,
        tol=1e-3):
    """Block Coordinate Descent

    Parameters
    ----------
    f
    g
    x0

    Returns
    -------

    """

    if x0 is None:
        xk = np.zeros(f.n_features)
    else:
        xk = np.array(x0, copy=True)
    if g is None:
        g = utils.ZeroLoss()
    if step_size is None:
        step_size = 1. / f.lipschitz_constant('features')

    Ax = f.A.dot(xk)
    f_alpha = f.alpha
    n_samples, n_features = f.A.shape
    success = False

    partial_gradient = f.partial_gradient_factory()
    prox = g.prox_factory()

    if trace:
        trace_x = np.zeros((max_iter, n_features))
    else:
        trace_x = np.zeros((0, 0))

    @njit(nogil=True)
    def _bcd_algorithm(
            x, Ax, A_csr_data, A_csr_indices, A_csr_indptr, A_csc_data,
            A_csc_indices, A_csc_indptr, b, trace_x, tol):
        it = 0
        feature_indices = np.arange(n_features)
        for it in range(1, max_iter):
            np.random.shuffle(feature_indices)
            for j in feature_indices:
                grad_j = 0.
                for i_indptr in range(A_csc_indptr[j], A_csc_indptr[j+1]):
                    # get the current sample
                    i_idx = A_csc_indices[i_indptr]
                    grad_j += partial_gradient(Ax[i_idx], b[i_idx]) * A_csc_data[i_indptr] / n_samples
                x_new = prox(x[j] - step_size * (grad_j + f_alpha * x[j]), step_size)

                # .. update Ax ..
                for i_indptr in range(A_csc_indptr[j], A_csc_indptr[j+1]):
                    i_idx = A_csc_indices[i_indptr]
                    Ax[i_idx] += A_csc_data[i_indptr] * (x_new - x[j])
                x[j] = x_new
                if j == 0:
                    # .. recompute Ax TODO: do only in async ..
                    for i in range(n_samples):
                        p = 0.
                        for j in range(A_csr_indptr[i], A_csr_indptr[i+1]):
                            j_idx = A_csr_indices[j]
                            p += x[j_idx] * A_csr_data[j]
                        # .. copy back to shared memory ..
                        Ax[i] = p

        return it, None

    X_csc = sparse.csc_matrix(f.A.A)
    X_csr = sparse.csr_matrix(f.A.A)

    trace_func = []
    start = datetime.now()
    trace_time = [(start - datetime.now()).total_seconds()]


    start_time = datetime.now()
    n_iter, certificate = _bcd_algorithm(
                xk, Ax, X_csr.data, X_csr.indices, X_csr.indptr, X_csc.data,
                X_csc.indices, X_csc.indptr, f.b, trace_x, tol)
    delta = (datetime.now() - start_time).total_seconds()

    if trace:
        trace_time = np.linspace(0, delta, n_iter)
        if verbose:
            print('Computing trace')
        # .. compute function values ..
        trace_func = []
        for i in range(n_iter):
            # TODO: could be parallelized
            trace_func.append(f(trace_x[i]) + g(trace_x[i]))
    return optimize.OptimizeResult(
        x=xk, success=success, nit=n_iter, trace_func=trace_func, trace_time=trace_time,
        certificate=certificate)



def minimize_SVRGLP_L1(
        f_deriv, A, b, x0, alpha=0, beta=0, step_size=None,
        max_iter=500, tol=1e-6, verbose=False, callback=None):
    """Stochastic average gradient augmented (SAGA) algorithm.

    The SAGA algorithm can solve optimization problems of the form

        argmin_{x \in R^p} \sum_{i}^n_samples f(A_i^T x, b_i) + alpha * ||x||_2^2 +
                                            + beta * ||x||_1

    Parameters
    ----------
    f, g
        loss functions. g can be none

    x0: np.ndarray or None, optional
        Starting point for optimization.

    step_size: float or None, optional
        Step size for the optimization. If None is given, this will be
        estimated from the function f.

    n_jobs: int
        Number of threads to use in the optimization. A number higher than 1
        will use the Asynchronous SAGA optimization method described in
        [Pedregosa et al., 2017]

    max_iter: int
        Maximum number of passes through the data in the optimization.

    tol: float
        Tolerance criterion. The algorithm will stop whenever the norm of the
        gradient mapping (generalization of the gradient for nonsmooth optimization)
        is below tol.

    verbose: bool
        Verbosity level. True might print some messages.

    trace: bool
        Whether to trace convergence of the function, useful for plotting and/or
        debugging. If ye, the result will have extra members trace_func,
        trace_time.

    Returns
    -------
    opt: OptimizeResult
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.

    References
    ----------
    The SAGA algorithm was originally described in

        Aaron Defazio, Francis Bach, and Simon Lacoste-Julien. `SAGA: A fast
        incremental gradient method with support for non-strongly convex composite
        objectives. <https://arxiv.org/abs/1407.0202>`_ Advances in Neural
        Information Processing Systems. 2014.

    The implemented has some improvements with respect to the original version, such as
    better support for sparse datasets and is described in

        Fabian Pedregosa, Remi Leblond, and Simon Lacoste-Julien. "Breaking the Nonsmooth
        Barrier: A Scalable Parallel Method for Composite Optimization." Advances in
        Neural Information Processing Systems (NIPS) 2017.
    """
    # convert any input to CSR sparse matrix representation. In the future we might want to
    # implement also a version for dense data (numpy arrays) to better exploit data locality
    x = np.ascontiguousarray(x0).copy()
    n_samples, n_features = A.shape

    if step_size is None:
        # then need to use line search
        raise ValueError

    # .. estimate diagonal elements of the reweighting matrix (D) ..
    tmp = A.copy()
    tmp.data[:] = 1.
    d = np.array(tmp.sum(0), dtype=np.float).ravel()
    idx = (d != 0)
    d[idx] = n_samples / d[idx]
    d[~idx] = 0.

    epoch_iteration, full_grad = _factory_SVRG_epoch(A, b, f_deriv, d, alpha, beta)

    # .. initialize memory terms ..
    idx = np.arange(n_samples)
    success = False
    nit = 0
    if callback is not None:
        callback(x)
    for nit in range(max_iter):
        x_snapshot = x.copy()
        gradient_average = full_grad(x_snapshot)
        np.random.shuffle(idx)
        # if callback is not None:
        #     callback(x)
        epoch_iteration(
            x, x_snapshot, idx, gradient_average, step_size)
        if callback is not None:
            callback(x)

        if np.abs(x - x_snapshot).sum() < tol:
            success = True
            break
        print(nit, np.linalg.norm(x - x_snapshot))
    message = ''
    return optimize.OptimizeResult(
        x=x, success=success, nit=nit,
        message=message)


def _factory_SVRG_epoch(A, b, f_deriv, d, alpha, beta, line_search=False):

    if beta > 0:
        @njit
        def prox(x, ss):
            return np.fmax(x - beta * ss, 0) - np.fmax(- x - beta * ss, 0)
    elif beta == 0:
        @njit
        def prox(x, ss):
            return x
    else:
        raise ValueError

    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    n_samples, n_features = A.shape

    @njit
    def full_grad(x):
        grad = np.zeros(x.size)
        for i in range(n_samples):
            p = 0.
            for j in range(A_indptr[i], A_indptr[i + 1]):
                j_idx = A_indices[j]
                p += x[j_idx] * A_data[j]
            grad_i = f_deriv(p, b[i])
            # .. gradient estimate (XXX difference) ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                grad[j_idx] += grad_i * A_data[j] / n_samples
        return grad

    @njit(nogil=True)
    def _svrg_algorithm(
            x, x_snapshot, idx, gradient_average, step_size):

        # .. inner iteration ..
        for i in idx:
            p = 0.
            p_old = 0.
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                p += x[j_idx] * A_data[j]
                p_old += x_snapshot[j_idx] * A_data[j]

            grad_i = f_deriv(p, b[i])
            old_grad = f_deriv(p_old, b[i])

            # .. update coefficients ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                delta = (grad_i - old_grad) * A_data[j]
                incr = delta + d[j_idx] * (gradient_average[j_idx] + alpha * x[j_idx])
                x[j_idx] = prox(x[j_idx] - step_size * incr, step_size * d[j_idx])

    return _svrg_algorithm, full_grad
