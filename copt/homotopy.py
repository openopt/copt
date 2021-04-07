"""Homotopy Methods. Currently, Homotopy Conditional Gradient Methods (a.k.a. Frank-Wolfe)"""
#%%

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

# TODO XXX remove this
np.random.seed(0)

EPS = np.finfo(np.float32).eps

# https://arxiv.org/pdf/1804.08544.pdf

def minimize_homotopy_cgm(objective_fun, smoothed_constraints, x0, lmo, beta0, max_iter, tol, callback):
    # TODO is this necessary?
    x0 = np.asanyarray(x0, dtype=np.float)
    if tol < 0:
        raise ValueError("'tol' must be non-negative")
    x = x0.copy()
    
    for it in range(max_iter):
        step_size = 2. / (it+2)
        beta_k = beta0 / np.sqrt(it+2)

        g_beta_t_grad = np.zeros(x.shape, dtype=np.float)
        for c in smoothed_constraints:
            # TODO remove this beta argument since maybe it is more numerically
            # stable to move it around (and avoid double divisions.)
            g_beta_t, constraint_grad = c.smoothed_g_grad(x, 1.)
            g_beta_t_grad += constraint_grad

        f_t, f_grad = objective_fun(x)
        grad = beta_k*f_grad + g_beta_t_grad

        active_set = None # vanilla FW
        update_direction, _, _, _ = lmo(-grad, x, active_set)
        # TODO what is this update direction normalization???
        # norm_update_direction = linalg.norm(update_direction)**2
        norm_update_direction = update_direction

        x += step_size*update_direction

        if callback is not None:
            if callback(locals()) is False:  # pylint: disable=g-bool-id-comparison
                break

    if callback is not None:
        callback(locals())

    # TODO return something

from pathlib import Path
DATA_DIR = os.environ.get(
    "COPT_DATA_DIR", os.path.join(os.path.expanduser("~"), "copt_data")
)
def reduced_digits():
    mat = sio.loadmat(Path(DATA_DIR) / "sdp_mnist" / "reduced_clustering_mnist_digits.mat")
    C = mat['Problem']['C'][0][0]
    opt_val = mat['Problem']['opt_val'][0][0][0][0]
    return C, opt_val

def full_digits():
    mat = sio.loadmat(Path(DATA_DIR) / "sdp_mnist" / "full_clustering_mnist_digits.mat")
    C = mat['Problem']['C'][0][0]
    opt_val = mat['Problem']['opt_val'][0][0][0][0]
    return C, opt_val

if True:
    C_mat, opt_val = reduced_digits()
else:
    C_mat, opt_val = full_digits()

class LinearObjective:
    def __init__(self, M):
        # TODO just flatten from the outset
        self.M = M

    def __call__(self, X):
        loss,_ = self.f_grad(X)
        return loss

    def f_grad(self, X):
        '''X is the decision variable.'''
        loss = np.dot(X.flatten(), self.M.flatten())
        grad = self.M
        grad = grad.flatten()
        return loss, grad

    @property
    def lipschitz(self):
        # TODO think
        _,D,_ = np.linalg.svd(self.M)
        largest_eigv = max(D)
        return largest_eigv / self.M.shape[0]

class RowEqualityConstraint:
    def __init__(self, shape, operator, offset):
        self.shape = shape
        # TODO "operator is vague"
        self.operator = operator
        self.offset = offset

    def __call__(self, x):
        X = x.reshape(self.shape)
        z = np.matmul(X, self.operator)
        return np.all(z == self.offset)

    def feasibility_dist_squared(self, x):
        X = x.reshape(self.shape)
        z = np.matmul(X, self.operator)
        return np.sum((z-self.offset) ** 2)

    def relative_feasibility_dist_squared(self, x):
        return self.feasibility_dist_squared(x)/linalg.norm(self.offset)

    def smoothed(self, x, beta):
        return .5/beta * self.feasibility_dist_squared(x)

    def smoothed_g_grad(self, x, beta):
        X = x.reshape(self.shape)
        z = np.matmul(X, self.operator)
        grad = 1./beta * np.outer((z-self.offset), self.offset)

        g_beta_val = self.smoothed(x, beta)
        return g_beta_val, grad.flatten()

class ElementWiseInequalityConstraint:
    def __init__(self, shape, offset):
        self.shape = shape
        self.offset = offset

    def __call__(self, x):
        return np.all(x >= self.offset)

    def feasibility_dist_squared(self, x):
        infeasible_vals = x[(x - self.offset) < 0]
        return np.sum(infeasible_vals**2)

    def relative_feasibility_dist_squared(self, x):
        # We don't want to count this in our empirical analysis
        return 0

    def smoothed(self, x, beta):
        return .5/beta * self.feasibility_dist_squared(x)

    def smoothed_g_grad(self, x, beta):
        # return self.smoothed(x, beta), 1000*np.minimum(x-self.offset, 0)
        return self.smoothed(x, beta), np.minimum(x-self.offset, 0)

# TODO remove
# TODO frequency
# TODO stats outfile
checkpoint_dir = Path(".")
stats_file = open(checkpoint_dir / 'stats.txt', 'a', buffering=1) # TODO What about closing this file?
class TraceFoo(copt.utils.Trace):
    def __init__(self, f=None, freq=1):
        super(TraceFoo, self).__init__(f, freq)
        self.trace_relative_subopt = []
        self.trace_feasibility_dist = []

    def __call__(self, dl):
        # TODO move all of this into the plotting code. Only trace the most
        # fundamental quantities that are needed to compute things later.

        # TODO trace with frequency < 1

        # TODO trace time as well?
        x = dl['x']
        f_t = dl['f_t']
        smoothed_constraints = dl['smoothed_constraints']
        relative_subopt = np.abs(f_t-opt_val)/opt_val
        total_feasibility_dist = sum(c.relative_feasibility_dist_squared(x) \
            for c in smoothed_constraints)

        self.trace_relative_subopt.append(relative_subopt)
        self.trace_feasibility_dist.append(total_feasibility_dist)

        it = dl['it']
        if it % 100 == 0:
            stats = dict(it=it, relative_subopt=relative_subopt, feasibility_dist=total_feasibility_dist)
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)

if False:
    linear_objective = LinearObjective(C_mat)

    sum_to_one_row_constraint = RowEqualityConstraint(C_mat.shape,
                                                    np.ones(C_mat.shape[1]),
                                                    np.ones(C_mat.shape[1]))

    non_negativity_constraint = ElementWiseInequalityConstraint(C_mat.shape, 0)

    # cb = copt.utils.Trace(linear_objective)
    cb = TraceFoo()

    n_labels = 10 # TODO (since it's MNIST)
    alpha = n_labels
    traceball = copt.constraint.TraceBall(alpha, C_mat.shape)
    x_init = np.zeros(C_mat.shape).flatten()
    beta0 = 1.

    minimize_homotopy_cgm(
        linear_objective.f_grad,
        [sum_to_one_row_constraint, non_negativity_constraint],
        x_init,
        traceball.lmo,
        beta0,
        tol = 0,
        callback=cb,
        max_iter=int(5e3)
    )

    if True:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.suptitle('Homotopy Frank-Wolfe')

        cb.trace_relative_subopt
        cb.trace_feasibility_dist

        ax1.plot(cb.trace_relative_subopt, label='Homotopy FW')
        ax1.grid(True)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylabel('relative suboptimality')

        ax2.plot(cb.trace_feasibility_dist)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_ylabel('feasibility convergence')
        ax2.grid(True)
    else:
        fig, ax = plt.subplots()
        fig.suptitle('Homotopy Frank-Wolfe')
        ax.plot(cb.trace_relative_subopt, label='Homotopy FW')
        ax.grid(True)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('relative suboptimality')


    if True:
        plt.show()
    else:
        plt.savefig("mnist_experiment.pdf")

# %%

def test_linear_objective():
    # TODO dependency on C_mat
    linear_objective = LinearObjective(C_mat)
    cb = copt.utils.Trace(linear_objective)
    alpha = 3.
    traceball = copt.constraint.TraceBall(alpha, C_mat.shape)

    x_init = np.zeros(C_mat.shape)
    x_init = x_init.flatten()

    sol = copt.minimize_frank_wolfe(
        linear_objective.f_grad,
        x_init,
        traceball.lmo,
        tol=0,
        lipschitz=linear_objective.lipschitz,
        callback=cb,
        step="sublinear",
        max_iter=1000,
    )

    def check(x):
        ss = 1 / linear_objective.lipschitz
        _,grad = linear_objective.f_grad(x)
        # this is the proximal mapping, zero at optimum
        grad_map = (x - traceball.prox(x - ss * grad, ss)) / ss
        return np.linalg.norm(grad_map)

    assert check(x_init) > 0.4
    assert check(sol.x) < 0.4

test_linear_objective()

# def test_row_equality_constraints():
#     # TODO set C_mat randomly
#     sum_to_one_row_constraint = RowEqualityConstraint(C_mat.shape,
#                                                   np.ones(C_mat.shape[1]),
#                                                   np.ones(C_mat.shape[1]))

#     beta = 1.
#     x = np.zeros(C_mat.shape)
#     x = x.flatten()
#     sum_to_one_row_constraint.smoothed(x, beta)

#     val, grad = sum_to_one_row_constraint.smoothed_g_grad(x, beta)

#     t = 1e-6
#     # vec = np.random.rand(*grad.shape)
#     vec = np.zeros(grad.shape)
#     vec[0] = 1
#     v1, _ = sum_to_one_row_constraint.smoothed_g_grad(x + t*vec, beta)
#     v2, _ = sum_to_one_row_constraint.smoothed_g_grad(x - t*vec, beta)

#     print(
#         np.linalg.norm(
#             (v1-v2)/(2*t) -
#             np.dot(grad, vec) * vec
#         ))

def test_row_equality_constraints():
    # TODO set C_mat randomly
    sum_to_one_row_constraint = RowEqualityConstraint(C_mat.shape,
                                                  np.ones(C_mat.shape[1]),
                                                  np.ones(C_mat.shape[1]))

    beta = 1.
    x = np.zeros(C_mat.shape)
    x = x.flatten()

    def f(x):
        val, _ = sum_to_one_row_constraint.smoothed_g_grad(x, beta)
        return val

    def g(x):
        _, grad = sum_to_one_row_constraint.smoothed_g_grad(x, beta)
        return grad

    from scipy import optimize

    for _ in range(100):
        print(optimize.check_grad(f, g, np.random.randn(*x.shape)))

#     val, grad = sum_to_one_row_constraint.smoothed_g_grad(x, beta)


test_row_equality_constraints()

