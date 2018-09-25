from datetime import datetime
import numpy as np
from scipy import sparse, optimize
from numba import njit
from copt import utils
from tqdm import trange


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
    if p > 0:
        return np.log(1 + np.exp(-p)) + (1 - y) * p
    else:
        return np.log(1 + np.exp(p)) - p * y


@njit
def deriv_logistic(p, y):
    # derivative of logistic loss
    # same as in lightning (with minus sign)
    if p > 0:
        tmp = np.exp(-p)
        phi = - tmp / (1. + tmp) + 1 - y
    else:
        tmp = np.exp(p)
        phi = tmp / (1. + tmp) - y
    return phi



def minimize_SAGA_L1(
        f_deriv, A, b, x0, step_size, alpha=0, beta=0,
        max_iter=500, tol=1e-6, verbose=1, callback=None):
    """Stochastic average gradient augmented (SAGA) algorithm for linearly-parametrized
    loss functions.

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
    A = sparse.csr_matrix(A)
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
    pbar = trange(max_iter, disable=(verbose == 0))
    for it in pbar:
        x_old = x.copy()
        np.random.shuffle(idx)
        _saga_epoch(
                x, idx, memory_gradient, gradient_average, step_size)
        if callback is not None:
            callback(x)

        diff_norm = np.abs(x - x_old).sum()
        if diff_norm < tol:
            success = True
            break
        if it % 10 == 0:
            pbar.set_description('SAGA iter %i' % it)
            pbar.set_postfix(tol=diff_norm)
    return optimize.OptimizeResult(
        x=x, success=success, nit=it)


def minimize_VRTOS(
        f_deriv, A, b, x0, step_size, prox_1=None, prox_2=None, blocks_1=None,
        blocks_2=None, alpha=0, beta=0, max_iter=500, tol=1e-6, callback=None,
        verbose=0):
    """
    TODO description
    """
    
    n_samples, n_features = A.shape
    success = False

    Y = np.zeros((2, x0.size))
    x = x0.copy()
    z = x.copy()

    assert A.shape[0] == b.size

    if step_size < 0:
        raise ValueError
    
    if prox_1 is None:
        @njit
        def prox_1(x, step_size):
            return x
    if prox_2 is None:
        @njit
        def prox_2(x, step_size):
            return x

    if blocks_1 is None:
        blocks_1 = np.arange(x.size)
    if blocks_2 is None:
        blocks_2 = np.arange(x.size)        
    for blocks in [blocks_1, blocks_2]:
        if np.any(np.diff(blocks) < 0):
            raise ValueError('blocks cannot be discontinuous nor with decreasing id')
            
    A = sparse.csr_matrix(A)
    epoch_iteration = _factory_sparse_VRSAGA(
        f_deriv, prox_1, prox_2, blocks_1, blocks_2, A, b,
        alpha, beta, step_size)

    # .. memory terms ..
    memory_gradient = np.zeros(n_samples)
    gradient_average = np.zeros(n_features)
    x1 = x0.copy()

    # warm up for the JIT
    epoch_iteration(
        Y, x0, x1, z, memory_gradient, gradient_average, np.array([0]),
        step_size)

    trace_func = []
    start_time = datetime.now()
    trace_time = [0.]
    trace_x = [x.copy()]
    trace_certificate = [np.inf]

    # .. iterate on epochs ..
    pbar = trange(max_iter, disable=(verbose == 0))
    for it in pbar:
        epoch_iteration(
            Y, x0, x1, z, memory_gradient, gradient_average, np.random.permutation(n_samples),
            step_size)

        certificate = np.linalg.norm(x0 - z)
        if callback is not None:
            callback(x)

        print(it,certificate)
        if it % 10 == 0:
            pbar.set_description('VRTOS iter %i' % it)

        if callback is not None:
            callback(x)

    return optimize.OptimizeResult(
        x=x0, success=success, nit=it, trace_x=trace_x,
        certificate=certificate,
        trace_func=trace_func, trace_certificate=np.array(trace_certificate),
        trace_time=trace_time)



@njit(nogil=True, cache=True)
def _support_matrix(
        A_indices, A_indptr, blocks, n_blocks):
    """
    Parameters
    ----------
    A_indices, A_indptr: numpy arrays representing the data matrix in CSR format.
    
    blocks: numy array of size n_features with integer values, where the value codes for the group to which the given feature belongs.
    
    n_blocks: number of unique blocks in array blocks.
    
    
    Returns
    -------
    Parameters of a CSR matrix representing the extended support. The returned vectors represent a sparse matrix of shape (n_samples, n_blocks), element (i, j) is one if j is in the extended support of f_i, zero otherwise.
    """
    if n_blocks == 1:
        # FIXME do something smart
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
            g_idx = blocks[A_indices[j]]
            if seen_blocks[g_idx] == 0:
                # if first time we encouter this block,
                # add to the index and mark as seen
                BS_indices[counter_indptr] = g_idx
                seen_blocks[g_idx] = 1
                counter_indptr += 1
        BS_indptr[i + 1] = counter_indptr
        # cleanup
        for j in range(BS_indptr[i], counter_indptr):
            seen_blocks[BS_indices[j]] = 0
    BS_data = np.ones(counter_indptr)
    return BS_data, BS_indices[:counter_indptr], BS_indptr


#@njit(nogil=True, cache=True)
def _csr_blocks(blocks, n_blocks):
    indices = np.arange(blocks.size, dtype=np.int)
    indptr = np.zeros(n_blocks+1, dtype=np.int)
    
    largest_seen_block = 0
    seen_blocks = 0
    for i in range(blocks.size):
        if blocks[i] > largest_seen_block:
            # jump
            indptr[seen_blocks + 1] = i
            pointer = blocks[i]
            seen_blocks += 1
    indptr[n_blocks] = i+1
    return indices, indptr



def _factory_sparse_VRSAGA(
        f_prime, prox_1, prox_2, blocks_1, blocks_2, A, b, alpha, beta, gamma):

    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    n_samples, n_features = A.shape

    unique_blocks_1 = np.unique(blocks_1)
    n_blocks_1 = np.unique(blocks_1).size
    assert np.all(unique_blocks_1 == np.arange(n_blocks_1))

    blocks_1_data, b1_indices, b1_indptr = _support_matrix(
        A_indices, A_indptr, blocks_1, n_blocks_1)
    csr_blocks_1 = sparse.csr_matrix((blocks_1_data, b1_indices, b1_indptr))

    unique_blocks_2 = np.unique(blocks_2)
    n_blocks_2 = np.unique(blocks_2).size
    assert np.all(unique_blocks_2 == np.arange(n_blocks_2))

    blocks_2_data, blocks_2_indices, b2_indptr = _support_matrix(
        A_indices, A_indptr, blocks_2, n_blocks_2)
    csr_blocks_2 = sparse.csr_matrix((blocks_2_data, blocks_2_indices, b2_indptr))

    # .. diagonal reweighting ..
    d1 = np.array(csr_blocks_1.sum(0), dtype=np.float).ravel()
    idx = (d1 != 0)
    d1[idx] = n_samples / d1[idx]
    d1[~idx] = 1
    
    d2 = np.array(csr_blocks_2.sum(0), dtype=np.float).ravel()
    idx = (d2 != 0)
    d2[idx] = n_samples / d2[idx]
    d2[~idx] = 1

    # .. XXX TODO description ..
    # This overwrites 
    b1r_indices, b1r_indptr = _csr_blocks(blocks_1, n_blocks_1)
    # csr_blocks_1 = sparse.csr_matrix((blocks_1, b1_indices, b1_indptr))

    b2r_indices, b2r_indptr = _csr_blocks(blocks_2, n_blocks_2)
    # csr_blocks_2 = sparse.csr_matrix((blocks_2, b2r_indices, b2_indptr))

    #@njit
    def epoch_iteration_template(
            Y, X1, X2, z, memory_gradient, gradient_average, sample_indices, step_size):

        # .. SAGA estimate of the gradient ..
        grad_est = np.zeros(n_features)
        z = np.zeros(n_features)

        # .. iterate on samples ..
        for i in sample_indices:
            
            # Compute z
            # do it naively for now, improve later
            for j in range(n_features):
                z[j] = (d1[blocks_1[j]] * X1[j] + d2[blocks_2[j]] * X2[j]) / (d1[blocks_1[j]] + d2[blocks_2[j]])

            p = 0.
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                p += z[j_idx] * A_data[j]

            grad_i = f_prime(p, b[i])

            # .. gradient estimate (XXX difference) ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                grad_est[j_idx] = (grad_i - memory_gradient[i]) * A_data[j]

            # .. iterate on blocks ..
            for h_j in range(b1_indptr[i], b1_indptr[i+1]):
                h = b1_indices[h_j]

                # .. iterate on features inside block ..
                for b_j in range(b1r_indptr[h], b1r_indptr[h+1]):
                    bias_term = d1[h] * (gradient_average[b_j] + alpha * z[b_j])
                    X1[b_j] = 2 * z[b_j] - Y[0, b_j] - step_size * 0.5 * (
                        grad_est[b_j] + bias_term)

                ss = d1[h] * step_size * beta
                X1[b1_indptr[h]:b1_indptr[h+1]] = prox_1(
                    X1[b1_indptr[h]:b1_indptr[h+1]], ss)

                # .. update y ..
                for b_j in range(b1r_indptr[h], b1r_indptr[h+1]):
                    Y[0, b_j] += X1[b_j] - z[b_j]


            for h_j in range(b2_indptr[i], b2_indptr[i+1]):
                h = blocks_2_indices[h_j]

                # .. iterate on features inside block ..
                for b_j in range(b2r_indptr[h], b2r_indptr[h+1]):
                    bias_term = d2[h] * (gradient_average[b_j] + alpha * z[b_j])
                    X2[b_j] = 2 * z[b_j] - Y[1, b_j] - step_size * 0.5 * (
                        grad_est[b_j] + bias_term)

                ss = d2[h] * step_size * beta
                X2[b2_indptr[h]:b2_indptr[h+1]] = prox_2(
                    X2[b2_indptr[h]:b2_indptr[h+1]], ss)


                # .. update y ..
                for b_j in range(b2r_indptr[h], b2r_indptr[h+1]):
                    Y[1, b_j] += X2[b_j] - z[b_j]

            # .. update memory terms ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                gradient_average[j_idx] += (grad_i - memory_gradient[i]) * A_data[j] / n_samples
                grad_est[j_idx] = 0
            memory_gradient[i] = grad_i

    return epoch_iteration_template



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
