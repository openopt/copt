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


@njit(nogil=True)
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


def prox_l1(alpha):
    @njit(nogil=True)
    def _prox_l1(x, ss):
        return np.fmax(x - alpha * ss, 0) - np.fmax(- x - alpha * ss, 0)
    return _prox_l1


def prox_gl(alpha):
    @njit(nogil=True)
    def _prox_gl(x, ss):
        norm = np.linalg.norm(x)
        if norm > alpha * ss:
            return (1 - alpha * ss / norm) * x
        else:
            return np.zeros_like(x)
    return _prox_gl


def minimize_SAGA_L1(
        f_deriv, A, b, x0, step_size, alpha=0, beta=0,
        max_iter=500, tol=1e-6, verbose=1, callback=None):
    """Stochastic average gradient augmented (SAGA) algorithm for linearly-parametrized
    loss functions.

    The SAGA algorithm can solve optimization problems of the form

        minimize_x \sum_{i}^n_samples f(A_i^T x, b_i) + alpha ||x||_2^2 + beta ||x||_1

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
        prox = prox_l1(beta)
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
    pbar.set_description('SAGA')
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
        pbar.set_postfix(tol=diff_norm)
    return optimize.OptimizeResult(
        x=x, success=success, nit=it)



def minimize_SVRG_L1(
        f_deriv, A, b, x0, step_size, alpha=0, beta=0, 
        max_iter=500, tol=1e-6, verbose=False, callback=None):
    """Stochastic average gradient augmented (SAGA) algorithm.

    The SAGA algorithm can solve optimization problems of the form

        argmin_{x \in R^p} \sum_{i}^n_samples f(A_i^T x, b_i) + alpha * ||x||_2^2 +
                                            + beta * ||x||_1

    Parameters
    ----------
    f_deriv
        derivative of f

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
    A = sparse.csr_matrix(A)

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


def minimize_VRTOS(
        f_deriv, A, b, x0, step_size, prox_1=None, prox_2=None, alpha=0, max_iter=500, tol=1e-6, callback=None,
        verbose=0):
    """Variance-reduced three operator splitting (VRTOS) algorithm.

    The VRTOS algorithm can solve optimization problems of the form

        argmin_{x \in R^p} \sum_{i}^n_samples f(A_i^T x, b_i) + alpha * ||x||_2^2 +
                                            + pen1(x) + pen2(x)

    Parameters
    ----------
    f_deriv
        derivative of f

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
    Pedregosa, Fabian, Kilian Fatras, and Mattia Casotto. "Variance Reduced Three Operator Splitting." arXiv preprint arXiv:1806.07294 (2018).
    """
    
    n_samples, n_features = A.shape
    success = False
    
    # FIXME: just a workaround for now
    # FIXME: check if prox_1 is a tuple
    if hasattr(prox_1, '__len__') and len(prox_1) == 2:
        blocks_1 = prox_1[1]
        prox_1 = prox_1[0]
    else:
        blocks_1 = sparse.eye(n_features, n_features, format='csr')
    if hasattr(prox_2, '__len__') and len(prox_2) == 2:
        blocks_2 = prox_2[1]
        prox_2 = prox_2[0]
    else:
        blocks_2 = sparse.eye(n_features, n_features, format='csr')

    Y = np.zeros((2, x0.size))
    z = x0.copy()

    assert A.shape[0] == b.size

    if step_size < 0:
        raise ValueError
    
    if prox_1 is None:
        @njit
        def prox_1(x, i, indices, indptr, d, step_size):
            pass
    if prox_2 is None:
        @njit
        def prox_2(x, i, indices, indptr, d, step_size):
            pass

    A = sparse.csr_matrix(A)
    epoch_iteration = _factory_sparse_VRTOS(
        f_deriv, prox_1, prox_2, blocks_1, blocks_2, A, b,
        alpha, step_size)

    # .. memory terms ..
    memory_gradient = np.zeros(n_samples)
    gradient_average = np.zeros(n_features)
    x1 = x0.copy()
    grad_tmp = np.zeros(n_features)

    # warm up for the JIT
    epoch_iteration(
        Y, x0, x1, z, memory_gradient, gradient_average, np.array([0]),
        grad_tmp, step_size)

    # .. iterate on epochs ..
    if callback is not None:
        callback(z)
    pbar = trange(max_iter, disable=(verbose == 0))
    for it in pbar:
        epoch_iteration(
            Y, x0, x1, z, memory_gradient, gradient_average,
            np.random.permutation(n_samples),
            grad_tmp, step_size)

        certificate = np.linalg.norm(x0 - z) + np.linalg.norm(x1 - z)
        if callback is not None:
            callback(z)

        pbar.set_description('VRTOS')
        pbar.set_postfix(tol=certificate)

    return optimize.OptimizeResult(
        x=z, success=success, nit=it,
        certificate=certificate)


@njit(nogil=True)
def _support_matrix(
        A_indices, A_indptr, reverse_blocks_indices, n_blocks):
    """
    Parameters
    ----------
    A_indices, A_indptr: numpy arrays representing the data matrix in CSR format.
    
    XXX changed blocks
    
    blocks: numy array of size n_features with integer values, where the value codes for the group to which the given feature belongs.
    
    n_blocks: number of unique blocks in array blocks.


    Notes
    -----
    BS stands for Block Support

    Returns
    -------
    Parameters of a CSR matrix representing the extended support. The returned
    vectors represent a sparse matrix of shape (n_samples, n_blocks), element (i, j) is one if j is in the extended support of f_i, zero otherwise.
    """
    BS_indices = np.zeros(A_indices.size, dtype=np.int64)
    BS_indptr = np.zeros(A_indptr.size, dtype=np.int64)
    seen_blocks = np.zeros(n_blocks, dtype=np.int64)
    BS_indptr[0] = 0
    counter_indptr = 0
    for i in range(A_indptr.size - 1):
        low = A_indptr[i]
        high = A_indptr[i + 1]
        for j in range(low, high):
            g_idx = reverse_blocks_indices[A_indices[j]]
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


def _factory_sparse_VRTOS(
        f_prime, prox_1, prox_2, blocks_1, blocks_2, A, b, alpha, gamma):

    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    n_samples, n_features = A.shape

    blocks_1_indptr = blocks_1.indptr
    blocks_2_indptr = blocks_2.indptr

    rblocks_1_indices = blocks_1.T.tocsr().indices
    bs_1_data, bs_1_indices, bs_1_indptr = _support_matrix(
        A_indices, A_indptr, rblocks_1_indices, blocks_1.shape[0])
    csr_blocks_1 = sparse.csr_matrix((bs_1_data, bs_1_indices, bs_1_indptr))

    rblocks_2_indices = blocks_2.T.tocsr().indices
    bs_2_data, bs_2_indices, bs_2_indptr = _support_matrix(
        A_indices, A_indptr, rblocks_2_indices, blocks_2.shape[0])
    csr_blocks_2 = sparse.csr_matrix((bs_2_data, bs_2_indices, bs_2_indptr))

    # .. diagonal reweighting ..
    d1 = np.array(csr_blocks_1.sum(0), dtype=np.float).ravel()
    idx = (d1 != 0)
    d1[idx] = n_samples / d1[idx]
    d1[~idx] = 1

    d2 = np.array(csr_blocks_2.sum(0), dtype=np.float).ravel()
    idx = (d2 != 0)
    d2[idx] = n_samples / d2[idx]
    d2[~idx] = 1

    @njit(nogil=True)
    def epoch_iteration_template(
            Y, x1, x2, z, memory_gradient, gradient_average, sample_indices,
            grad_tmp, step_size):

        # .. iterate on samples ..
        for i in sample_indices:
            p = 0.
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                p += z[j_idx] * A_data[j]

            grad_i = f_prime(p, b[i])

            # .. gradient estimate ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                grad_tmp[j_idx] = (grad_i - memory_gradient[i]) * A_data[j]

            # .. x update ..
            for h_j in range(bs_1_indptr[i], bs_1_indptr[i+1]):
                h = bs_1_indices[h_j]

                # .. iterate on features inside block ..
                for b_j in range(blocks_1_indptr[h], blocks_1_indptr[h+1]):
                    bias_term = d1[h] * (gradient_average[b_j] + alpha*z[b_j])
                    x1[b_j] = 2 * z[b_j] - Y[0, b_j] - step_size * 0.5 * (
                        grad_tmp[b_j] + bias_term)

            prox_1(x1, i, bs_1_indices, bs_1_indptr, d1, step_size)

            # .. update y ..
            for h_j in range(bs_1_indptr[i], bs_1_indptr[i+1]):
                h = bs_1_indices[h_j]
                for b_j in range(blocks_1_indptr[h], blocks_1_indptr[h+1]):
                    Y[0, b_j] += x1[b_j] - z[b_j]

            for h_j in range(bs_2_indptr[i], bs_2_indptr[i+1]):
                h = bs_2_indices[h_j]

                # .. iterate on features inside block ..
                for b_j in range(blocks_2_indptr[h], blocks_2_indptr[h+1]):
                    bias_term = d2[h] * (gradient_average[b_j] + alpha * z[b_j])
                    x2[b_j] = 2 * z[b_j] - Y[1, b_j] - step_size * 0.5 * (
                        grad_tmp[b_j] + bias_term)

            prox_2(x2, i, bs_2_indices, bs_2_indptr, d2, step_size)

            # .. update y ..
            for h_j in range(bs_2_indptr[i], bs_2_indptr[i+1]):
                h = bs_2_indices[h_j]
                for b_j in range(blocks_2_indptr[h], blocks_2_indptr[h+1]):
                    Y[1, b_j] += x2[b_j] - z[b_j]
            
            # .. update z ..
            for h_j in range(bs_1_indptr[i], bs_1_indptr[i+1]):
                h = bs_1_indices[h_j]
            
                # .. iterate on features inside block ..
                for b_j in range(blocks_1_indptr[h], blocks_1_indptr[h+1]):
                    da = 1./d1[rblocks_1_indices[b_j]]
                    db = 1./d2[rblocks_2_indices[b_j]]
                    z[b_j] = (da * Y[0, b_j] + db * Y[1, b_j]) / (da + db)

            for h_j in range(bs_2_indptr[i], bs_2_indptr[i+1]):
                h = bs_2_indices[h_j]

                # .. iterate on features inside block ..
                for b_j in range(blocks_2_indptr[h], blocks_2_indptr[h+1]):
                    da = 1./d1[rblocks_1_indices[b_j]]
                    db = 1./d2[rblocks_2_indices[b_j]]
                    z[b_j] = (da * Y[0, b_j] + db * Y[1, b_j]) / (da + db)

            # .. update memory terms ..
            for j in range(A_indptr[i], A_indptr[i+1]):
                j_idx = A_indices[j]
                gradient_average[j_idx] += (grad_i - memory_gradient[i]) * A_data[j] / n_samples
                grad_tmp[j_idx] = 0
            memory_gradient[i] = grad_i

    return epoch_iteration_template
