import numpy as np
import warnings
from scipy import sparse, optimize
from numba import njit


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


def saga(A, b, x0, loss, stepsize, max_iter=1000):
    """
    The SAGA algorithm, for solving an optimization problem of the form

        argmin_x \frac{1}{n} \sum_{i=1}^n f(a_i^T x, b_i)

    Parameters
    ----------
    f_prime: callable
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

    """
    #
    # if stepsize is None:
    #     # compute stepsize
    #     stepsize = get_auto_step_size(A, alpha, loss, None) / 4

    x = np.ascontiguousarray(x0).copy()

    if loss == 'log':
        f_prime = deriv_logistic
    elif loss == 'squared':
        f_prime = deriv_squared
    else:
        raise NotImplementedError

    n_samples, n_features = A.shape

    epoch_iteration = _epoch_iteration_factory(f_prime)

    # initialize variables
    memory_gradient = np.zeros(n_samples)  # TODO: set to f_i'(0)

    sample_indices = np.arange(n_samples)

    # temporary storage, perhaps could be avoided
    gradient_average = np.zeros(n_features)

    # iterate on epochs
    for it in range(max_iter):
        # permute samples (maybe a full shuffle of X and y
        # would be more efficient)
        np.random.shuffle(sample_indices)
        epoch_iteration(
            A, b, x, memory_gradient, gradient_average, sample_indices, stepsize)
        # if callback is not None:
        #     callback(z)
    success = True  # XXX
    return optimize.OptimizeResult(
        x=x, success=success, nit=it)


def _epoch_iteration_factory(f_prime):

    @njit
    def epoch_iteration_template(A, b, x, memory_gradient, gradient_average, sample_indices,
                                 step_size):
        n_samples, n_features = A.shape
        # inner iteration
        for i in sample_indices:
            grad_i = f_prime(x, A[i], b[i])
            x -= step_size * ((grad_i - memory_gradient[i]) * A[i] + gradient_average)
            gradient_average += A[i] * (grad_i - memory_gradient[i]) / n_samples
            memory_gradient[i] = grad_i
    return epoch_iteration_template

