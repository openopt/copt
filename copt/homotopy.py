"""Homotopy Methods."""
import json
import os
import sys
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as sio
from scipy import linalg, optimize

import copt
from copt import datasets, utils

EPS = np.finfo(np.float32).eps

# TODO :ref:`hcgm`
def minimize_homotopy_cgm(objective_fun, smoothed_constraints, x0, shape,
                          lmo, beta0, max_iter, tol, callback):
    r"""(H)omotopy CGM

    Implements HCGM. See :ref:`hcgm` for more details

    Args:
        objective_fun: callable
          Takes an x0-like and return a tuple (value, gradient) corresponding
          to the value and gradient of the objective function at x,
          respectively.

        smoothed_constraints: [list of obj]
          Each object should support a `smoothed_grad` method. This method
          takes an x0-like and returns a tuple (value, gradient)
          corresponding to the value of the smoothed constraint at x and the
          smoothed gradient, respectively.

        x0: array-like
          Initial guess for solution.

        shape: tuple
          The underlying shape of each iterate, e.g. (n,m) for an n x m matrix.
        
        beta0: float
          Initial value for the smoothing parameter beta.

        max_iter: integer, optional
          Maximum number of iterations.

        tol: float
          Tolerance for the objective and homotopy smoothed constraints.

        callback: callable, optional
          Callback to execute at each iteration. If the callable returns False
          then the algorithm with immediately return.

        lmo: callable

    Returns:
        scipy.optimize.OptimizeResult
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.

    References:
        ..
        [YURT2018] A. Yurtsever, O. Fercoq, F. Locatello, and V. Cevher. “A Conditional Gradient Framework for Composite Convex Minimization with Applications to Semidefinite Programming” <http://arxiv.org/abs/1804.08544> _ ICML 2018 
    """

    x0 = np.asanyarray(x0, dtype=np.float)
    beta0 = np.asanyarray(beta0, dtype=np.float128)
    if tol < 0:
        raise ValueError("'tol' must be non-negative")
    x = x0.copy()
    
    for it in range(max_iter):
        step_size = 2. / (it+2.)
        beta_k = beta0 / np.sqrt(it+2)

        total_constraint_grad = sum(c.smoothed_grad(x)[1] for c in smoothed_constraints)
        f_t, f_grad = objective_fun(x)
        grad = beta_k*f_grad + total_constraint_grad

        # symmetrize gradient. This can be beneficial if the LMO is slightly
        # less accurate (e.g. svd instead of eig solver)
        grad_square = grad.reshape(shape)
        grad_square = .5*(grad_square + grad_square.T)
        grad = grad_square.flatten()

        active_set = None # vanilla FW
        update_direction, _, _, _ = lmo(-grad, x, active_set)

        feasibilities = [c(x) < tol for c in smoothed_constraints]
        if f_t < tol and np.all(feasibilities):
            break

        x += step_size*update_direction

        if callback is not None:
            if callback(locals()) is False:  # pylint: disable=g-bool-id-comparison
                break

    if callback is not None:
        callback(locals())

    return optimize.OptimizeResult(x=x, nit=it)
