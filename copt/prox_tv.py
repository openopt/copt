# Authors: Fabian Pedregosa based on the code of Laurent Condat
#

"""
These are some helper functions to compute the proximal operator of some common penalties
"""

import numpy as np
from numba import njit
import warnings


def prox_tv1d(w, step_size):
    """
    Computes the proximal operator of the 1-dimensional total variation operator.

    This solves a problem of the form

         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2

    where TV(x) is the one-dimensional total variation

    Parameters
    ----------
    w: array
        vector of coefficients
    stepsize: float
        step size (sometimes denoted gamma) in proximal objective function

    References
    ----------
    Condat, Laurent. "A direct algorithm for 1D total variation denoising."
    IEEE Signal Processing Letters (2013)
    """

    if w.dtype not in (np.float32, np.float64):
        raise ValueError('argument w must be array of floats')
    return _prox_tv1d(w, step_size, 1)


@njit
def _prox_tv1d(w, step_size, incr):
    """low level function call, no checks are performed"""
    output = w.copy()
    j = 0
    width = w.size + 1
    index_low = np.zeros(width, dtype=np.int32)
    slope_low = np.zeros(width, dtype=w.dtype)
    index_up  = np.zeros(width, dtype=np.int32)
    slope_up  = np.zeros(width, dtype=w.dtype)
    index     = np.zeros(width, dtype=np.int32)
    z         = np.zeros(width, dtype=w.dtype)
    y_low     = np.empty(width, dtype=w.dtype)
    y_up      = np.empty(width, dtype=w.dtype)
    s_low, c_low, s_up, c_up, c = 0, 0, 0, 0, 0
    y_low[0] = y_up[0] = 0
    y_low[1] = w[0] - step_size
    y_up[1] = w[0] + step_size

    for i in range(2, width):
        y_low[i] = y_low[i-1] + w[(i - 1) * incr]
        y_up[i] = y_up[i-1] + w[(i - 1) * incr]

    y_low[width-1] += step_size
    y_up[width-1] -= step_size
    slope_low[0] = np.inf
    slope_up[0] = -np.inf
    z[0] = y_low[0]

    for i in range(1, width):
        c_low += 1
        c_up += 1
        index_low[c_low] = index_up[c_up] = i
        slope_low[c_low] = y_low[i]-y_low[i-1]
        while (c_low > s_low+1) and (slope_low[max(s_low, c_low-1)] <= slope_low[c_low]):
            c_low -= 1
            index_low[c_low] = i
            if c_low > s_low+1:
                slope_low[c_low] = (y_low[i]-y_low[index_low[c_low-1]]) / (i-index_low[c_low-1])
            else:
                slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])

        slope_up[c_up] = y_up[i]-y_up[i-1]
        while (c_up > s_up+1) and (slope_up[max(c_up-1, s_up)] >= slope_up[c_up]):
            c_up -= 1
            index_up[c_up] = i
            if c_up > s_up + 1:
                slope_up[c_up] = (y_up[i]-y_up[index_up[c_up-1]]) / (i-index_up[c_up-1])
            else:
                slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

        while (c_low == s_low+1) and (c_up > s_up+1) and (slope_low[c_low] >= slope_up[s_up+1]):
            c += 1
            s_up += 1
            index[c] = index_up[s_up]
            z[c] = y_up[index[c]]
            index_low[s_low] = index[c]
            slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])
        while (c_up == s_up+1) and (c_low>s_low+1) and (slope_up[c_up]<=slope_low[s_low+1]):
            c += 1
            s_low += 1
            index[c] = index_low[s_low]
            z[c] = y_low[index[c]]
            index_up[s_up] = index[c]
            slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

    for i in range(1, c_low - s_low + 1):
        index[c+i] = index_low[s_low+i]
        z[c+i] = y_low[index[c+i]]
    c = c + c_low-s_low
    i = 1
    while i <= c:
        a = (z[i]-z[i-1]) / (index[i]-index[i-1])
        while j < index[i]:
            output[j * incr] = a
            output[j * incr] = a
            j += 1
        i += 1
    return output


@njit
def prox_tv1d_cols(a, stepsize, n_rows, n_cols):
    """

    Parameters
    ----------
    a
    stepsize
    n_rows
    n_cols

    Returns
    -------

    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_cols):
        out[:, i] = _prox_tv1d(A[:, i], stepsize, 1)
    return out.ravel()


@njit
def prox_tv1d_rows(a, stepsize, n_rows, n_cols):
    """
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_rows):
        out[i] = _prox_tv1d(A[i, :], stepsize, 1)
    return out.ravel()


def c_prox_tv2d(x, stepsize, n_rows, n_cols, max_iter, tol):
    """
    Douflas-Rachford to minimize a 2-dimensional total variation.

    Reference: https://arxiv.org/abs/1411.0589
    """
    n_features = n_rows * n_cols
    p = np.zeros(n_features)
    q = np.zeros(n_features)

    for it in range(max_iter):
        y = x + p
        y = prox_tv1d_cols(y, stepsize, n_rows, n_cols)
        p += x - y
        x = y + q
        x = prox_tv1d_rows(x, stepsize, n_rows, n_cols)
        q += y - x

        # check convergence
        accuracy = np.max(np.abs(y - x))
        if accuracy < tol:
            break
    else:
        warnings.warn("prox_tv2d did not converged to desired accuracy\n" +
                      "Accuracy reached: %s" % accuracy)
    return x


def prox_tv2d(w, stepsize, n_rows, n_cols, max_iter=500, tol=1e-3):
    """
    Computes the proximal operator of the 2-dimensional total variation operator.

    This solves a problem of the form

         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2

    where TV(x) is the two-dimensional total variation. It does so using the
    Douglas-Rachford algorithm [Barbero and Sra, 2014].

    Parameters
    ----------
    w: array
        vector of coefficients

    stepsize: float
        step size (often denoted gamma) in proximal objective function

    max_iter: int

    tol: float

    References
    ----------
    Condat, Laurent. "A direct algorithm for 1D total variation denoising."
    IEEE Signal Processing Letters (2013)

    Barbero, Alvaro, and Suvrit Sra. "Modular proximal optimization for
    multidimensional total-variation regularization." arXiv preprint
    arXiv:1411.0589 (2014).
    """

    x = w.copy().astype(np.float64)
    return c_prox_tv2d(x, stepsize, n_rows, n_cols, max_iter, tol)
