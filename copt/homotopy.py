"""Homotopy Methods. Currently, Homotopy Conditional Gradient Methods (a.k.a. Frank-Wolfe)"""
import os, sys
import warnings
import json
from collections import defaultdict
import numpy as np
from scipy import linalg
from scipy import optimize
from scipy import io as sio
import copt
from copt import utils
from copt import datasets
import matplotlib.pyplot as plt

EPS = np.finfo(np.float32).eps

# https://arxiv.org/pdf/1804.08544.pdf

def minimize_homotopy_cgm(objective_fun, smoothed_constraints, x0, shape, lmo, beta0, max_iter, tol, callback):
    # TODO is this necessary?
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
        grad_square = grad.reshape(shape) # TODO hard coded
        grad_square = .5*(grad_square + grad_square.T)
        grad = grad_square.flatten()

        active_set = None # vanilla FW
        update_direction, _, _, _ = lmo(-grad, x, active_set)

        x += step_size*update_direction

        if callback is not None:
            if callback(locals()) is False:  # pylint: disable=g-bool-id-comparison
                break

    if callback is not None:
        callback(locals())

    # TODO return something