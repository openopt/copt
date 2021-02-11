
"""Homotopy Methods. Currently, Homotopy Conditional Gradient Methods (a.k.a. Frank-Wolfe)"""
import warnings
from collections import defaultdict
import numpy as np
from scipy import linalg
from scipy import optimize
from copt import utils


EPS = np.finfo(np.float32).eps


# https://arxiv.org/pdf/1804.08544.pdf

def minimize_homotopy_cgm(objective_fun, smoothed_constraint_fun, x0, lmo, beta0, max_iter, tol, callback):
    pass



# Need to setup the experiment here. Get the dataset described in the paper and go from there

