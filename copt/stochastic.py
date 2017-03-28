import concurrent.futures
from datetime import datetime
import numpy as np
from scipy import sparse, optimize
from numba import njit
from copt import utils


def minimize_SAGA(
    f, g=None, x0=None, step_size=None, n_jobs: int=1, max_iter=500,
    tol=1e-6, verbose=False, trace=False) -> optimize.OptimizeResult:
    """Stochastic average gradient augmented (SAGA) algorithm.

    The SAGA algorithm can solve optimization problems of the form

                       argmin_x f(x) + g(x)

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
        [Leblond et al., 2017]

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
        Aaron Defazio, Francis Bach, and Simon Lacoste-Julien. `SAGA: A fast
        incremental gradient method with support for non-strongly convex composite
        objectives. <https://arxiv.org/abs/1407.0202>`_ Advances in Neural
        Information Processing Systems. 2014.

        Rémi Leblond, Fabian Pedregosa, Simon Lacoste-Julien. `ASAGA: Asynchronous
        parallel SAGA <https://arxiv.org/abs/1606.04809>`_. Proceedings of the 20th
        International Conference on Artificial Intelligence and Statistics (AISTATS).
        2017.
    """
    if x0 is None:
        x = np.zeros(f.n_features)
    else:
        x = np.ascontiguousarray(x0).copy()

    if step_size is None:
        step_size = 1. / (3 * f.lipschitz_constant())

    if g is None:
        g = utils.ZeroLoss()

    success = False
    epoch_iteration = _factory_sparse_SAGA(f, g)
    n_samples, n_features = f.A.shape

    # .. memory terms ..
    memory_gradient = np.zeros(n_samples)
    gradient_average = np.zeros(n_features)

    if trace:
        trace_x = np.zeros((max_iter, n_features))
    else:
        trace_x = np.zeros((0, 0))
    stop_flag = np.zeros(1, dtype=np.bool)
    trace_func = []
    trace_time = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for job_id in range(n_jobs):
            futures.append(executor.submit(
                epoch_iteration, x, memory_gradient, gradient_average,
                step_size, max_iter, job_id, tol, stop_flag, trace, trace_x,
                np.random.permutation(n_samples), True))
        start_time = datetime.now()

        n_iter, certificate = futures[0].result()
        delta = (datetime.now() - start_time).total_seconds()
        concurrent.futures.wait(futures)

    if trace:
        trace_time = np.linspace(0, delta, n_iter)
        if verbose:
            print('Computing trace')
        # .. compute function values ..
        trace_func = []
        for i in range(n_iter):
            # TODO: could be parallelized
            trace_func.append(f(trace_x[i]) + g(trace_x[i]))

    if certificate < tol:
        success = True

    return optimize.OptimizeResult(
        x=x, success=success, nit=n_iter, trace_func=trace_func, trace_time=trace_time,
        certificate=certificate)


@njit(nogil=True)
def _support_matrix(
        A_indices, A_indptr, g_blocks, n_blocks):
    """
    """
    if n_blocks == 1:
        # XXX FIXME do something smart
        pass
    BS_indices = np.zeros(A_indices.size, dtype=np.int64)
    BS_indptr = np.zeros(A_indptr.size, dtype=np.int64)
    seen_blocks = np.zeros(n_blocks, dtype=np.int64)
    BS_indptr[0] = 0
    counter_indptr = 0
    for i in range(A_indptr.size - 1):
        low = A_indptr[i]
        high = A_indptr[i + 1]
        for j in range(low, high):
            g_idx = g_blocks[A_indices[j]]
            if seen_blocks[g_idx] == 0:
                # if first time we encouter this block,
                # add to the index and mark as seen
                BS_indices[counter_indptr] = g_idx
                seen_blocks[g_idx] = 1
                counter_indptr += 1
        BS_indptr[i+1] = counter_indptr
        # cleanup
        for j in range(BS_indptr[i], counter_indptr):
            seen_blocks[BS_indices[j]] = 0
    BS_data = np.ones(counter_indptr)
    return BS_data, BS_indices[:counter_indptr], BS_indptr


def _factory_sparse_SAGA(f, g):

    A = sparse.csr_matrix(f.A)
    b = f.b
    f_alpha = f.alpha
    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    n_samples, n_features = A.shape

    partial_gradient = f.partial_gradient_factory()
    prox = g.prox_factory()

    # .. compute the block support ..
    if g.is_separable:
        g_blocks = np.arange(n_features)
    else:
        raise NotImplementedError
    # g_blocks is a map from n_features -> n_features
    unique_blocks = np.unique(g_blocks)
    n_blocks = np.unique(g_blocks).size
    assert np.all(unique_blocks == np.arange(n_blocks))

    BS_data, BS_indices, BS_indptr = _support_matrix(
        A_indices, A_indptr, g_blocks, n_blocks)
    BS = sparse.csr_matrix((BS_data, BS_indices, BS_indptr), (n_samples, n_blocks))

    d = np.array(BS.sum(0), dtype=np.float).ravel()
    idx = (d != 0)
    d[idx] = n_samples / d[idx]
    d[~idx] = 0.

    @njit(nogil=True)
    def _saga_algorithm(
            x, memory_gradient, gradient_average, step_size, max_iter, job_id,
            tol, stop_flag, trace, trace_x, sample_indices, async):

        # .. SAGA estimate of the gradient ..
        x_old = x.copy()
        cert = np.inf
        it = 0

        if job_id == 0 and trace:
            trace_x[0, :] = x

        # .. inner iteration ..
        for it in range(1, max_iter):
            np.random.shuffle(sample_indices)
            for i in sample_indices:
                p = 0.
                for j in range(A_indptr[i], A_indptr[i+1]):
                    j_idx = A_indices[j]
                    p += x[j_idx] * A_data[j]

                grad_i = partial_gradient(p, b[i])

                # .. update coefficients ..
                for j in range(A_indptr[i], A_indptr[i+1]):
                    j_idx = A_indices[j]
                    incr = (grad_i - memory_gradient[i]) * A_data[j]
                    incr += d[j_idx] * (gradient_average[j_idx] + f_alpha * x[j_idx])
                    x[j_idx] = prox(x[j_idx] - step_size * incr, step_size * d[j_idx])

                # .. update memory terms ..
                for j in range(A_indptr[i], A_indptr[i+1]):
                    j_idx = A_indices[j]
                    gradient_average[j_idx] += (
                            grad_i - memory_gradient[i]) * A_data[j] / n_samples
                memory_gradient[i] = grad_i

                if i == 0 and async:
                    # .. recompute alpha bar ..
                    grad_tmp = np.zeros(n_features)
                    for i_inner in sample_indices:
                        for j in range(A_indptr[i_inner], A_indptr[i_inner + 1]):
                            j_idx = A_indices[j]
                            grad_tmp[j_idx] += memory_gradient[i_inner] * A_data[j] / n_samples
                    # .. copy back to shared memory ..
                    gradient_average[:] = grad_tmp

            if job_id == 0:
                if trace:
                    trace_x[it, :] = x
                # .. convergence check ..
                # cert = np.linalg.norm(x - x_old) / step_size
                # x_old[:] = x
                # if cert < tol:
                #     stop_flag[0] = True
                #     break
        #
        #     if stop_flag[0]:
        #         break
        #
        # # .. if any job has finished, stop the whole algorithm ..
        # stop_flag[0] = True
        return it, cert

    return _saga_algorithm


def minimize_BCD(
        f, g=None, x0=None, step_size=None, max_iter=500, trace=False, verbose=False,
        n_jobs=1):
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
        step_size = 2. / f.lipschitz_constant()

    Ax = f.A.dot(xk)
    f_alpha = f.alpha
    n_samples, n_features = f.A.shape
    success = False

    partial_gradient = f.partial_gradient_factory()
    prox = g.prox_factory()

    start_time = datetime.now()
    if trace:
        trace_x = np.zeros((max_iter, n_features))
    else:
        trace_x = np.zeros((0, 0))
    stop_flag = np.zeros(1, dtype=np.bool)

    @njit(nogil=True)
    def _bcd_algorithm(
            x, Ax, A_csr_data, A_csr_indices, A_csr_indptr, A_csc_data,
            A_csc_indices, A_csc_indptr, b, trace_x, job_id):
        feature_indices = np.arange(n_features)
        it = 0
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

            if job_id == 0:
                if trace:
                    trace_x[it, :] = x

        return it, None

    X_csc = sparse.csc_matrix(f.A)
    X_csr = sparse.csr_matrix(f.A)

    trace_func = []
    start = datetime.now()
    trace_time = [(start - datetime.now()).total_seconds()]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for job_id in range(n_jobs):
            futures.append(executor.submit(
                _bcd_algorithm,
                xk, Ax, X_csr.data, X_csr.indices, X_csr.indptr, X_csc.data,
                X_csc.indices, X_csc.indptr, f.b, trace_x, job_id))
        concurrent.futures.wait(futures)

    n_iter, certificate = futures[0].result()
    if trace:
        delta = (datetime.now() - start_time).total_seconds()
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

