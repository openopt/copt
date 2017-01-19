from concurrent import futures
from typing import Callable
from datetime import datetime
import numpy as np
from scipy import sparse, optimize
from numba import njit
from copt.utils import norm_rows


@njit
def f_squared(w, x, y):
    # squared loss
    return 0.5 * ((y - np.dot(x, w)) ** 2)


@njit
def deriv_squared(w, x, y):
    # derivative of squared loss
    return - (y - np.dot(x, w))


@njit
def f_logistic(w, x, y):
    # logistic loss
    # same as in lightning
    p = y * np.dot(x, w)
    if p > 0:
        return np.log(1 + np.exp(-p))
    else:
        return -p + np.log(1 + np.exp(p))


@njit
def deriv_logistic(w, x, y):
    # derivative of logistic loss
    # same as in lightning (with minus sign)
    p = y * np.dot(x, w)
    if p > 0:
        phi = 1. / (1 + np.exp(-p))
    else:
        exp_t = np.exp(p)
        phi = exp_t / (1. + exp_t)
    return (phi - 1) * y


def compute_step_size(loss: str, A, step_size_factor=4) -> float:
    """
    Helper function to compute the step size for common loss
    functions.

    Parameters
    ----------
    loss
    A
    step_size_factor

    Returns
    -------

    """
    if loss == 'logistic':
        return 4.0 / (norm_rows(A) * step_size_factor)
    elif loss == 'squared':
        return 1.0 / (norm_rows(A) * step_size_factor)
    else:
        raise NotImplementedError('loss %s is not implemented' % loss)


def fmin_SAGA(
        fun: Callable, fun_deriv: Callable, A, b, x0: np.ndarray,
        alpha: float=0., beta: float=0., g_prox: Callable=None, step_size: float=-1,
        g_blocks: np.ndarray=None, n_jobs: int=1, max_iter=100, tol=1e-6,
        verbose=False, callback=None, trace=False) -> optimize.OptimizeResult:
    """Stochastic average gradient augmented (SAGA) algorithm.

    The SAGA algorithm can solve optimization problems of the form

        argmin_x 1/n \sum_{i=1}^n f(a_i^T x, b_i) + alpha * L2 + beta * g(x)


    Parameters
    ----------
    fun
        loss function

    fun_deriv
        derivative function

    alpha
        Amount of squared L2 regularization

    x0
        Starting point

    g_blocks
        If g is a block-separable function, this allows to specify which are the
        blocks in this penalty. It is an array of integers with the same size as
        x0 where each coordinate represents the group to which that coordinate
        belongs to.

    Returns
    -------
    opt
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.

    References
    ----------
    Defazio, Aaron, Francis Bach, and Simon Lacoste-Julien. "SAGA: A fast
    incremental gradient method with support for non-strongly convex composite
    objectives." Advances in Neural Information Processing Systems. 2014.
    """

    x = np.ascontiguousarray(x0).copy()
    assert x.size == A.shape[1]
    assert A.shape[0] == b.size

    if step_size < 0:
        raise ValueError

    # TODO: encapsulate this in _get_factory
    if hasattr(g_prox, '__call__'):
        g_prox = njit(g_prox)
    # elif hasattr(g_prox, '__len__') and hasattr(g_prox, '__getitem__'):
    #     # its a list of proximal operators
    #     if not len(g_prox):
    #         raise NotImplementedError
    #     # TODO: make sure these are callable
    #     g_prox[0] = njit(g_prox[0])
    #     g_prox[1] = njit(g_prox[1])
    elif g_prox is None:
        @njit
        def g_prox(step_size, x, *args): return x
    else:
        raise NotImplementedError

    n_samples, n_features = A.shape
    success = False

    if sparse.issparse(A):
        A = sparse.csr_matrix(A)
        if g_blocks is None:
            g_blocks = np.arange(n_features)
        epoch_iteration, trace_loss = _epoch_factory_sparse_SAGA(
                fun, fun_deriv, g_prox, g_blocks, A, b, alpha, beta)
    else:
        epoch_iteration, trace_loss = _epoch_factory_SAGA(
            fun, fun_deriv, g_prox, A, b, alpha, beta)

    start_time = datetime.now()
    trace_fun = []
    trace_time = []
    trace_x = []

    # .. memory terms ..
    memory_gradient = np.zeros(n_samples)
    gradient_average = np.zeros(n_features)

    # .. iterate on epochs ..
    for it in range(max_iter):
        with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            fut = []
            for _ in range(n_jobs):
                fut.append(executor.submit(
                    epoch_iteration, x, memory_gradient, gradient_average,
                    np.random.permutation(n_samples), step_size))
            futures.wait(fut)
        if callback is not None:
            callback(x)
        if trace:
            trace_x.append(x.copy())
            trace_time.append((datetime.now() - start_time).total_seconds())

        # TODO: needs to be adapted in the sparse case
        grad = gradient_average + alpha * x
        certificate = np.linalg.norm(
            (x - g_prox(beta * step_size, x - step_size * grad)) / step_size)
        if verbose:
            print(it, certificate)
        if certificate < tol:
            success = True
            break
    if trace:
        if verbose:
            print('.. computing trace ..')
        # .. compute function values ..
        with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            trace_fun = [t for t in executor.map(trace_loss, trace_x)]

    return optimize.OptimizeResult(
        x=x, success=success, nit=it, trace_fun=trace_fun, trace_time=trace_time,
        certificate=certificate)


def fmin_PSSAGA(
        fun, fun_deriv, A, b, x0, g_prox=None, h_prox=None,
        alpha: float=0.0,
        beta: float=0.0,
        gamma: float=0.0,
        step_size=-1,
        g_func=None,
        h_func=None,
        g_blocks=None,
        h_blocks=None,
        max_iter=100, tol=1e-6, verbose=False, callback=None, trace=False):

    if hasattr(g_prox, '__call__'):
        if not hasattr(g_prox, 'inspect_llvm'):
            g_prox = njit(g_prox)
    elif g_prox is None:
        @njit
        def g_prox(step_size, x, *args): return x
    else:
        raise NotImplementedError

    if hasattr(h_prox, '__call__'):
        if not hasattr(h_prox, 'inspect_llvm'):
            # if it has not yet been jitted
            h_prox = njit(h_prox)
    elif h_prox is None:
        @njit
        def h_prox(step_size, x, *args): return x
    else:
        raise NotImplementedError

    y0 = x0.copy()
    y1 = x0.copy()
    # assert y.shape[1] == A.shape[1]
    assert A.shape[0] == b.size

    if step_size < 0:
        raise ValueError

    n_samples, n_features = A.shape
    success = False

    if sparse.issparse(A):
        A = sparse.csr_matrix(A)
        if h_blocks is None:
            h_blocks = np.arange(n_features)
        if g_blocks is None:
            g_blocks = np.arange(n_features)
        epoch_iteration, trace_loss = _epoch_factory_sparse_PSSAGA(
            fun, g_func, h_func, fun_deriv, g_prox, h_prox, g_blocks, h_blocks, A, b,
            alpha, beta, gamma)
    else:
        epoch_iteration, trace_loss = _epoch_factory_PSSAGA(
            fun, fun_deriv, g_prox, h_prox, A, b, alpha, beta, gamma)

    start_time = datetime.now()
    trace_func = []
    trace_certificate = []
    trace_time = []
    trace_x = []

    # .. memory terms ..
    memory_gradient = np.zeros(n_samples)
    gradient_average = np.zeros(n_features)
    x = y0.copy()
    z0 = x.copy()
    z1 = x.copy()

    # .. iterate on epochs ..
    for it in range(max_iter):
        epoch_iteration(
            y0, y1, x, z0, z1, memory_gradient, gradient_average, np.random.permutation(n_samples),
            step_size)

        xmz = np.concatenate((x - z0, x - z1)) / step_size
        certificate = np.linalg.norm(xmz)
        if callback is not None:
            callback(x)
        if trace:
            trace_x.append(x.copy())
            trace_certificate.append(certificate)
            trace_time.append((datetime.now() - start_time).total_seconds())
        if verbose:
            print('Iteration %s, certificate: %s' % (it, certificate))
        if certificate < tol:
            success = True
            break
    if trace:
        if verbose:
            print('.. computing trace ..')
        # .. compute function values ..
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            trace_func = [t for t in executor.map(trace_loss, trace_x)]

    return optimize.OptimizeResult(
        x=x, y=[y0, y1], success=success, nit=it, trace_x=trace_x,
        trace_func=trace_func, trace_certificate=np.array(trace_certificate),
        trace_time=trace_time)


def _epoch_factory_SAGA(fun, f_prime, g_prox, A, b, alpha, beta):

    @njit
    def epoch_iteration_template(
            x, memory_gradient, gradient_average, sample_indices,
            step_size):
        n_samples, n_features = A.shape
        # .. inner iteration ..
        for i in sample_indices:
            grad_i = f_prime(x, A[i], b[i])
            incr = (grad_i - memory_gradient[i]) * A[i]
            x[:] = g_prox(
                beta * step_size,
                x - step_size * (incr + gradient_average + alpha * x))
            gradient_average += incr / n_samples
            memory_gradient[i] = grad_i

    @njit
    def full_loss(x):
        obj = 0.
        n_samples, n_features = A.shape
        for i in range(n_samples):
            obj += fun(x, A[i], b[i]) / n_samples
        return obj + 0.5 * alpha * np.dot(x, x)

    return epoch_iteration_template, full_loss


@njit(nogil=True, cache=True)
def _support_matrix(
        A_indices, A_indptr, g_blocks, n_blocks):
    BS_indices = np.zeros(A_indices.size, dtype=np.int64)
    BS_indptr = np.zeros(A_indptr.size, dtype=np.int64)
    seen_blocks = np.zeros(n_blocks, dtype=np.bool_)
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


def _epoch_factory_sparse_SAGA(
        f_func, f_prime, g_prox, g_blocks, A, b, alpha, beta):

    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    n_samples, n_features = A.shape

    # g_blocks is a map from n_features -> n_features
    unique_blocks = np.unique(g_blocks)
    n_blocks = np.unique(g_blocks).size
    assert np.all(unique_blocks == np.arange(n_blocks))

    # .. compute the block support ..
    BS_data, BS_indices, BS_indptr = _support_matrix(
        A_indices, A_indptr, g_blocks, n_blocks)
    BS = sparse.csr_matrix((BS_data, BS_indices, BS_indptr), (n_samples, n_blocks))

    # .. estimate a mapping from blocks to features ..
    reverse_blocks = sparse.dok_matrix((n_blocks, n_features), dtype=np.bool)
    for j in range(n_features):
        i = g_blocks[j]
        reverse_blocks[i, j] = True
    reverse_blocks = reverse_blocks.tocsr()
    reverse_blocks_indices = reverse_blocks.indices
    reverse_blocks_indptr = reverse_blocks.indptr

    d = np.array(BS.sum(0), dtype=np.float).ravel()
    idx = (d != 0)
    d[idx] = n_samples / d[idx]

    @njit(nogil=True, cache=True)
    def epoch_iteration_template(
            x, memory_gradient, gradient_average, sample_indices, step_size):

        # .. SAGA estimate of the gradient ..
        grad_est = np.zeros(n_features)

        # .. inner iteration ..
        for i in sample_indices:
            idx = A_indices[A_indptr[i]:A_indptr[i+1]]
            block_idx = BS_indices[BS_indptr[i]:BS_indptr[i+1]]
            A_i = A_data[A_indptr[i]:A_indptr[i+1]]
            grad_i = f_prime(x[idx], A_i, b[i])

            # .. update coefficients ..
            grad_est[idx] = (grad_i - memory_gradient[i]) * A_i
            for g in block_idx:
                idx_g = reverse_blocks_indices[
                    reverse_blocks_indptr[g]:reverse_blocks_indptr[g+1]]
                grad_est[idx_g] += d[g] * (gradient_average[idx_g] + alpha * x[idx_g])
                x[idx_g] = g_prox(
                    step_size * beta * d[g], x[idx_g] - step_size * grad_est[idx_g])

                # .. clean up ..
                grad_est[idx_g] = 0

            # .. update memory terms ..
            gradient_average[idx] += (grad_i - memory_gradient[i]) * A_i / n_samples
            memory_gradient[i] = grad_i

    @njit(nogil=True, cache=True)
    def full_loss(x):
        obj = 0.
        for i in range(n_samples):
            idx = A_indices[A_indptr[i]:A_indptr[i + 1]]
            A_i = A_data[A_indptr[i]:A_indptr[i + 1]]
            obj += f_func(x[idx], A_i, b[i]) / n_samples
        return obj + 0.5 * alpha * np.dot(x, x)

    return epoch_iteration_template, full_loss


def _epoch_factory_sparse_PSSAGA(
        fun, g_func, h_func, f_prime, g_prox, h_prox, g_blocks, h_blocks, A, b, alpha,
        beta, gamma):

    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    n_samples, n_features = A.shape

    # g_blocks is a map from n_features -> n_features
    unique_blocks_g = np.unique(g_blocks)
    unique_blocks_h = np.unique(h_blocks)
    n_blocks_g = np.unique(g_blocks).size
    n_blocks_h = np.unique(h_blocks).size
    assert np.all(unique_blocks_g == np.arange(n_blocks_g))
    assert np.all(unique_blocks_h == np.arange(n_blocks_h))

    # .. compute the block support ..
    BS_g_data, BS_g_indices, BS_g_indptr = _support_matrix(
        A_indices, A_indptr, g_blocks, n_blocks_g)
    BS_g = sparse.csr_matrix((BS_g_data, BS_g_indices, BS_g_indptr))

    BS_h_data, BS_h_indices, BS_h_indptr = _support_matrix(
        A_indices, A_indptr, h_blocks, n_blocks_h)
    BS_h = sparse.csr_matrix((BS_h_data, BS_h_indices, BS_h_indptr))

    # .. estimate a mapping from blocks to features ..
    reverse_blocks_g = sparse.dok_matrix((n_blocks_g, n_features), dtype=np.bool)
    for j in range(n_features):
        i = g_blocks[j]
        reverse_blocks_g[i, j] = True
    reverse_blocks_g = reverse_blocks_g.tocsr()
    reverse_blocks_g_indices = reverse_blocks_g.indices
    reverse_blocks_g_indptr = reverse_blocks_g.indptr

    # .. estimate a mapping from blocks to features ..
    reverse_blocks_h = sparse.dok_matrix((n_blocks_h, n_features), dtype=np.bool)
    for j in range(n_features):
        i = h_blocks[j]
        reverse_blocks_h[i, j] = True
    reverse_blocks_h = reverse_blocks_h.tocsr()
    reverse_blocks_h_indices = reverse_blocks_h.indices
    reverse_blocks_h_indptr = reverse_blocks_h.indptr

    sparse_weights_g = np.array(BS_g.sum(0), dtype=np.float).ravel()
    idx = (sparse_weights_g != 0)
    sparse_weights_g[idx] = n_samples / sparse_weights_g[idx]

    sparse_weights_h = np.array(BS_h.sum(0), dtype=np.float).ravel()
    idx = sparse_weights_h != 0
    sparse_weights_h[idx] = n_samples / sparse_weights_h[idx]

    @njit(cache=True)
    def epoch_iteration_template(
            y0, y1, x, z0, z1, memory_gradient, gradient_average, sample_indices, step_size):

        # .. SAGA estimate of the gradient ..
        grad_est = np.zeros(n_features)
        x[:] = (y0 + y1) / 2.
        # .. inner iteration ..
        for i in sample_indices:
            idx = A_indices[A_indptr[i]:A_indptr[i+1]]
            block_g_idx = BS_g_indices[BS_g_indptr[i]:BS_g_indptr[i+1]]
            block_h_idx = BS_h_indices[BS_h_indptr[i]:BS_h_indptr[i+1]]
            A_i = A_data[A_indptr[i]:A_indptr[i+1]]
            grad_i = f_prime(x[idx], A_i, b[i])

            # .. update coefficients ..
            grad_est[idx] = (grad_i - memory_gradient[i]) * A_i
            for g in block_g_idx:
                idx_g = reverse_blocks_g_indices[
                    reverse_blocks_g_indptr[g]:reverse_blocks_g_indptr[g+1]]
                # Why is this in the opposite direction that you would
                # hope for?
                grad_est[idx_g] += (
                    gradient_average[idx_g] + alpha * x[idx_g]) * sparse_weights_g[g]
                z0[idx_g] = g_prox(
                    step_size * beta * sparse_weights_g[g],
                    2 * x[idx_g] - y0[idx_g] - 0.5 * step_size * grad_est[idx_g])
                y0[idx_g] = y0[idx_g] - x[idx_g] + z0[idx_g]

                # .. clean up ..
                grad_est[idx_g] = 0

            grad_est[idx] = (grad_i - memory_gradient[i]) * A_i
            for h in block_h_idx:
                idx_h = reverse_blocks_h_indices[
                    reverse_blocks_h_indptr[h]:reverse_blocks_h_indptr[h+1]]
                grad_est[idx_h] += (
                    gradient_average[idx_h] + alpha * x[idx_h]) * sparse_weights_h[h]
                z1[idx_h] = h_prox(
                    step_size * gamma * sparse_weights_h[h],
                    2 * x[idx_h] - y1[idx_h] - 0.5 * step_size * grad_est[idx_h])
                y1[idx_h] = y1[idx_h] - x[idx_h] + z1[idx_h]

                # .. clean up ..
                grad_est[idx_h] = 0

                # .. update x ..
                x[idx_h] = (y0[idx_h] + y1[idx_h]) / 2.

            for g in block_g_idx:
                # in theory only need to update those of x that
                # have still not been updated ...
                idx_g = reverse_blocks_g_indices[
                    reverse_blocks_g_indptr[g]:reverse_blocks_g_indptr[g+1]]
                x[idx_g] = (y0[idx_g] + y1[idx_g]) / 2.

            # .. update memory terms ..
            gradient_average[idx] += (grad_i - memory_gradient[i]) * A_i / n_samples
            memory_gradient[i] = grad_i
    @njit
    def full_loss(x):
        obj = 0.
        for i in range(n_samples):
            idx = A_indices[A_indptr[i]:A_indptr[i + 1]]
            A_i = A_data[A_indptr[i]:A_indptr[i + 1]]
            obj += fun(x[idx], A_i, b[i]) / n_samples
        obj += alpha * np.dot(x, x) + beta * g_func(x) + gamma * h_func(x)
        return obj

    return epoch_iteration_template, full_loss


def _epoch_factory_PSSAGA(fun, f_prime, g_prox, h_prox, A, b, alpha, beta, gamma):

    @njit
    def epoch_iteration_template(
            y, y1, x, z0, z1, memory_gradient, gradient_average, sample_indices,
            step_size):
        n_samples, n_features = A.shape
        # .. inner iteration ..
        for i in sample_indices:
            x[:] = g_prox(beta * step_size, y)
            grad_i = f_prime(x, A[i], b[i])
            incr = (grad_i - memory_gradient[i]) * A[i]
            z = h_prox(
                gamma * step_size,
                2 * x - y - step_size * (incr + gradient_average + alpha * x))
            y -= x - z
            gradient_average += incr / n_samples
            memory_gradient[i] = grad_i

    @njit
    def full_loss(x):
        obj = 0.
        n_samples, n_features = A.shape
        for i in range(n_samples):
            obj += fun(x, A[i], b[i]) / n_samples

    return epoch_iteration_template, full_loss