# Authors: Fabian Pedregosa
#

"""
These are some helper functions to compute the proximal operator of some common penalties
"""

import numpy as np
from numba import njit


def prox_tv1d(w, stepsize):
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
    tmp = w.copy()
    return c_prox_tv1d(tmp, stepsize)

@njit
def c_prox_tv1d(w, stepsize):
    incr = 1
    width = w.size

    # /to avoid invalid memory access to input[0] and invalid lambda values
    if width > 0 and stepsize >= 0:
        k, k0 = 0, 0			# k: current sample location, k0: beginning of current segment
        umin = stepsize  # u is the dual variable
        umax = - stepsize
        vmin = w[0] - stepsize
        vmax = w[0] + stepsize	  # bounds for the segment's value
        kplus = 0
        kminus = 0 	# last positions where umax=-lambda, umin=lambda, respectively
        twolambda = 2.0 * stepsize  # auxiliary variable
        minlambda = -stepsize		# auxiliary variable
        while True:				# simple loop, the exit test is inside
            while k >= width-1: 	# we use the right boundary condition
                if umin < 0.0:			# vmin is too high -> negative jump necessary
                    while True:
                        w[incr * k0] = vmin
                        k0 += 1
                        if k0 > kminus:
                            break
                    k = k0
                    kminus = k
                    vmin = w[incr * kminus]
                    umin = stepsize
                    umax = vmin + umin - vmax
                elif umax > 0.0:    # vmax is too low -> positive jump necessary
                    while True:
                        w[incr * k0] = vmax
                        k0 += 1
                        if k0 > kplus:
                            break
                    k = k0
                    kplus = k
                    vmax = w[incr * kplus]
                    umax = minlambda
                    umin = vmax + umax -vmin
                else:
                    vmin += umin / (k-k0+1)
                    while True:
                        w[incr * k0] = vmin
                        k0 += 1
                        if k0 > k:
                            break
                    return w
            umin += w[incr * (k + 1)] - vmin
            if umin < minlambda:       # negative jump necessary
                while True:
                    w[incr * k0] = vmin
                    k0 += 1
                    if k0 > kminus:
                        break
                k = k0
                kminus = k
                kplus = kminus
                vmin = w[incr * kplus]
                vmax = vmin + twolambda
                umin = stepsize
                umax = minlambda
            else:
                umax += w[incr * (k + 1)] - vmax
                if umax > stepsize:
                    while True:
                        w[incr * k0] = vmax
                        k0 += 1
                        if k0 > kplus:
                            break
                    k = k0
                    kminus = k
                    kplus = kminus
                    vmax = w[incr * kplus]
                    vmin = vmax - twolambda
                    umin = stepsize
                    umax = minlambda
                else:                   # no jump necessary, we continue
                    k += 1
                    if umin >= stepsize:		# update of vmin
                        kminus = k
                        vmin += (umin - stepsize) / (kminus - k0 + 1)
                        umin = stepsize
                    if umax <= minlambda:	    # update of vmax
                        kplus = k
                        vmax += (umax + stepsize) / (kplus - k0 + 1)
                        umax = minlambda
    return w


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
        out[:, i] = c_prox_tv1d(A[:, i], stepsize)
    return out.ravel()


@njit
def prox_tv1d_rows(a, stepsize, n_rows, n_cols):
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
    for i in range(n_rows):
        out[i] = c_prox_tv1d(A[i, :], stepsize)
    return out.ravel()


def c_prox_tv2d(x, stepsize, n_rows, n_cols, max_iter, tol):
    """
    Douflas-Rachford to minimize a 2-dimensional total variation.

    Reference: https://arxiv.org/abs/1411.0589
    """
    n_features = n_rows * n_cols
    p = np.zeros(n_features)
    q = np.zeros(n_features)

    # set X to the content of x
    for it in range(max_iter):
        y = x + p
        prox_tv1d_cols(y, stepsize, n_rows, n_cols)
        p += x - y
        x = y + q
        prox_tv1d_rows(x, stepsize, n_rows, n_cols)
        q += y - x

        # check convergence
        if np.max(np.abs(q - x)) < tol:
            continue
        else:
            break
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

    x = w.copy()
    return c_prox_tv2d(x, stepsize, n_rows, n_cols, max_iter, tol)
