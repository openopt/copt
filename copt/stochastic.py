from threading import Thread
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


def fmin_SAGA(
        fun, fun_deriv, A, b, x0, step_size=None, g_prox=None, n_jobs=1,
        max_iter=1000, tol=1e-6, verbose=True, callback=None, epochs_per_iter=5,
        trace=False):
    """Stochastic average gradient augumented (SAGA) algorithm.

    The SAGA algorithm can solve optimization problems of the form

        argmin_x \frac{1}{n} \sum_{i=1}^n f(a_i^T x, b_i) + g(x)

    Parameters
    ----------
    fun: callable or string
        XXX

    fun_deriv: callable or None
        f_prime(a_i^T x, b_i) returns the (scalar) derivative of f with
        respect to its first argument.

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
    Defazio, Aaron, Francis Bach, and Simon Lacoste-Julien. "SAGA: A fast
    incremental gradient method with support for non-strongly convex composite
    objectives." Advances in Neural Information Processing Systems. 2014.
    """

    x = np.ascontiguousarray(x0).copy()
    assert x.size == A.shape[1]
    assert A.shape[0] == b.size

    if fun == 'logistic':
        fun = f_logistic
        fun_deriv = deriv_logistic
        if step_size is None:
            max_rows = norm_rows(A)
            step_size = 4.0 / max_rows
    elif fun == 'squared':
        fun = f_squared
        fun_deriv = deriv_squared
        if step_size is None:
            max_rows = norm_rows(A)
            step_size = 1.0 / max_rows
    elif hasattr(fun, '__call__'):
        pass
    else:
        raise NotImplementedError

    n_samples, n_features = A.shape
    success = False

    if sparse.issparse(A):
        A = sparse.csr_matrix(A)
        epoch_iteration, full_loss = _epoch_factory_sparse(
            fun, fun_deriv, A, b)
    else:
        epoch_iteration, full_loss = _epoch_factory_dense(
            fun, fun_deriv, A, b)

    start_time = datetime.now()
    trace_fun = []
    trace_time = []

    # .. memory terms ..
    memory_gradient = np.zeros(n_samples)
    gradient_average = np.zeros(n_features)

    # .. iterate on epochs ..
    for it in range(max_iter):
        threads = []
        if trace:
            x2 = np.random.randn(n_features)
            t = Thread(target=full_loss, args=(x2, A.indices, A.indptr))
            threads.append(t)
        for _ in range(n_jobs):
            t = Thread(
                target=epoch_iteration,
                args=(x, memory_gradient, gradient_average,
                      np.random.permutation(n_samples), step_size))
            threads.append(t)

        # .. launch threads ..
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if callback is not None:
            callback(x)

        norm_grad = np.linalg.norm(gradient_average)
        if verbose:
            print(it, norm_grad)
        if norm_grad < tol:
            success = True
            break
    return optimize.OptimizeResult(
        x=x, success=success, nit=it, trace_fun=trace_fun, trace_time=trace_time)


def _epoch_factory_dense(fun, f_prime, A, b):

    @njit(nogil=True, cache=True)
    def epoch_iteration_template(
            x, memory_gradient, gradient_average, sample_indices,
            step_size):
        n_samples, n_features = A.shape
        # .. inner iteration ..
        for i in sample_indices:
            grad_i = f_prime(x, A[i], b[i])
            incr = (grad_i - memory_gradient[i]) * A[i]
            x -= step_size * (incr + gradient_average)
            gradient_average += incr / n_samples
            memory_gradient[i] = grad_i

    @njit(nogil=True, cache=True)
    def full_loss(x):
        obj = 0.
        n_samples, n_features = A.shape
        for i in range(n_samples):
            obj += fun(x, A[i], b[i]) / n_samples

    return epoch_iteration_template, full_loss


def _epoch_factory_sparse(fun, f_prime, A, b):

    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr
    n_samples, n_features = A.shape

    @njit
    def _debiasing_vec(A_indices, A_indptr):
        d = np.zeros(n_features)
        for i in range(n_samples):
            for j in A_indices[A_indptr[i]:A_indptr[i+1]]:
                d[j] += 1
        return n_samples / d

    d = _debiasing_vec(A_indices, A_indptr)

    @njit(nogil=True, cache=True)
    def epoch_iteration_template(
            x, memory_gradient, gradient_average, sample_indices, step_size):
        # .. inner iteration ..
        for i in sample_indices:
            idx = A_indices[A_indptr[i]:A_indptr[i+1]]
            A_i = A_data[A_indptr[i]:A_indptr[i+1]]
            grad_i = f_prime(x[idx], A_i, b[i])

            # .. update coefficients ..
            incr = (grad_i - memory_gradient[i]) * A_i
            x[idx] -= step_size * (incr + d[idx] * gradient_average[idx])

            # .. update memory terms ..
            gradient_average[idx] += incr / n_samples
            memory_gradient[i] = grad_i

    @njit(nogil=True, cache=True)
    def full_loss(x, A_indices, A_indptr):
        obj = 0.
        for i in range(n_samples):
            idx = A_indices[A_indptr[i]:A_indptr[i + 1]]
            A_i = A_data[A_indptr[i]:A_indptr[i + 1]]
            # obj += fun(x[idx], A_i, b[i]) / n_samples

    return epoch_iteration_template, full_loss

