"""Stochastic Homotopy Methods."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, optimize

import copt
from copt import datasets, utils

EPS = np.finfo(np.float32).eps

def sample_batches_with_replacement(length, batch_size):
    while True:
        yield np.random.choice(length, batch_size, replace=False)

def minimize_sag_homotopy_cgm(objective_fun, smoothed_constraints, x0, shape,
                          lmo, beta0, max_iter, tol, callback):
    r"""SAG (H)omotopy CGM

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
    TODO
    """

    x0 = np.asanyarray(x0, dtype=np.float)
    beta0 = np.asanyarray(beta0, dtype=np.float128)
    if tol < 0:
        raise ValueError("'tol' must be non-negative")
    x = x0.copy()

    # TODO hard coded
    bs_percent = .05
    # bs_percent = 1.
    # batch_sizes = (int(shape[0]*bs_percent), x.shape[0])
    batch_sizes = (int(shape[0]*bs_percent), int(shape[0]**2*bs_percent)) 

    constraint_batches = [sample_batches_with_replacement(c.n_constraints, bs) for (c, bs)
                          in zip(smoothed_constraints, batch_sizes)]

    constraint_grad_estimation = [np.zeros(c.n_constraints) for c 
                                  in smoothed_constraints]

    for it in range(max_iter):
        step_size = 2. / (it+2.)
        beta_k = beta0 / np.sqrt(it+2)

        f_t, f_grad = objective_fun(x)

        # update gradient estimations and then map from constraint space to
        # weight space
        total_constraint_grad = np.zeros(x.shape)
        for i in range(len(smoothed_constraints)):
            batch = next(constraint_batches[i])
            c = smoothed_constraints[i]
            gamma = constraint_grad_estimation[i]
            partial_grad, batch = c.partial_smoothed_grad(x, batch)
            gamma[batch] = partial_grad
            total_constraint_grad += c.apply_adjoint(gamma)

        grad = beta_k*f_grad + total_constraint_grad

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
