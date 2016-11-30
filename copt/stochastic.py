import numpy as np
import warnings
from scipy import sparse
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
    if p > 18:
        return np.exp(-p)
    if p < -18:
        return -p
    else:
        return np.log(1 + np.exp(-p))


@njit
def deriv_logistic(w, x, y):
    # derivative of logistic loss
    # same as in lightning (with minus sign)
    z = y * np.dot(x, w)
    if z > 18.0:
        return -np.exp(-z) * y
    if z < -18.0:
        return -y
    return -y / (np.exp(z) + 1.0)


def saga(A, b, loss):
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

    """

    if stepsize is None:
        # compute stepsize
        stepsize = get_auto_step_size(A, alpha, loss, None) / 4

    if loss == 'log':
        f_prime = deriv_logistic
    elif loss == 'squared':
        f_prime = deriv_squared
    else:
        raise NotImplementedError

    if penalty1 == 'l1':
        prox1 = prox_l1
    elif hasattr(penalty1, '__call__'):
        prox1 = penalty1
    else:
        raise NotImplementedError

    if penalty2 == 'l1':
        prox2 = prox_l1
    elif hasattr(penalty2, '__call__'):
        prox2 = penalty2
    else:
        raise NotImplementedError

    n_samples, n_features = A.shape

    epoch_iteration = epoch_iteration_factory(
        f_prime, prox1, prox2)

    # initialize variables
    z = np.zeros((2, n_features))
    memory_gradient = np.zeros(n_samples)  # TODO: set to f_i'(0)

    sample_indices = np.arange(n_samples)

    # temporary storage, perhaps could be avoided
    tmp = np.zeros((2, n_features))
    gradient_average = np.zeros(n_features)

    # iterate on epochs
    for it in range(max_iter):
        # permute samples (maybe a full shuffle of X and y
        # would be more efficient)
        np.random.shuffle(sample_indices)
        epoch_iteration(
            A, b, z, memory_gradient, gradient_average, sample_indices, alpha,
            beta1, beta2, stepsize, tmp, n_features, n_samples)
        if callback is not None:
            callback(z)
    return np.mean(z, axis=0)


def epoch_iteration_factory(f_prime, prox1, prox2):

    @njit
    def epoch_iteration_template(X, y, z, memory_gradient, gradient_average, sample_indices,
                                 alpha, beta1, beta2, stepsize, tmp, n_features, n_samples):
        k = 2
        # inner iteration
        for i in sample_indices:
            z_bar = ax0_mean(z)

            grad_i = f_prime(z_bar, X[i], y[i])

            # gradient step
            for j in range(n_features):
                g_ij = X[i, j] * (grad_i - memory_gradient[i]) + gradient_average[j] + \
                        alpha * z_bar[j]
                tmp[0, j] = 2 * z_bar[j] - z[0, j] - stepsize * g_ij
                tmp[1, j] = 2 * z_bar[j] - z[1, j] - stepsize * g_ij

            # prox step
            tmp[0] = prox1(tmp[0], k * beta1 * stepsize)
            tmp[1] = prox2(tmp[1], k * beta2 * stepsize)

            for j in range(n_features):
                z[0, j] = z[0, j] - z_bar[j] + tmp[0, j]
                z[1, j] = z[1, j] - z_bar[j] + tmp[1, j]

            for j in range(n_features):
                gradient_average[j] += X[i, j] * (grad_i-memory_gradient[i]) / n_samples

            memory_gradient[i] = grad_i
    return epoch_iteration_template


@njit
def ax0_mean(X):
    "assume it is a 2d array"
    n_samples, n_features = X.shape
    out = np.zeros((n_features,))

    for j in range(n_features):
        for i in range(n_samples):
            out[j] += X[i, j]
        out[j] /= n_samples
    return out