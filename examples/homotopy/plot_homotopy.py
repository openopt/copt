import copt
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import json

from copt.loss import LinearObjective
from copt.constraint import RowEqualityConstraint, ElementWiseInequalityConstraint
from copt.homotopy import minimize_homotopy_cgm

from pathlib import Path
# TODO remove this
basepath = Path("/Users/gideon/projects/openopt-copt/examples/homotopy")
def reduced_digits():
    mat = sio.loadmat(basepath / "data/reduced_clustering_mnist_digits.mat")
    C = mat['Problem']['C'][0][0]
    opt_val = mat['Problem']['opt_val'][0][0][0][0]
    labels = mat['Problem']['labels'][0][0][0]
    n_labels = len(np.unique(labels))
    return C, n_labels, opt_val

def full_digits():
    mat = sio.loadmat(basepath / "data/full_clustering_mnist_digits.mat")
    C = mat['Problem']['C'][0][0]
    opt_val = mat['Problem']['opt_val'][0][0][0][0]
    labels = mat['Problem']['labels'][0][0][0]
    n_labels = len(np.unique(labels))
    return C, n_labels, opt_val

class HomotopyTrace(copt.utils.Trace):
    def __init__(self, stats_filehandle=None, f=None, freq=1):
        super(HomotopyTrace, self).__init__(f, freq)
        self.trace_relative_subopt = []
        self.trace_feasibilities = []
        self.statsfile = stats_filehandle

    def __call__(self, dl):
        it = dl['it']
        x = dl['x']
        f_t = dl['f_t']
        smoothed_constraints = dl['smoothed_constraints']

        relative_subopt = np.abs(f_t-opt_val)/opt_val
        self.trace_relative_subopt.append(relative_subopt)

        stats = dict(it=it, objective=relative_subopt)
        self.trace_feasibilities.append([])
        for c in smoothed_constraints:
            feasibility = c.feasibility(x)
            stats[c.name] = feasibility
            self.trace_feasibilities[-1].append((c.name, feasibility))

        if self.statsfile is not None:
            print(json.dumps(stats), file=self.statsfile)

        if it % 100 == 0:
            print(json.dumps(stats))

if True:
    C_mat, n_labels, opt_val = reduced_digits()
else:
    C_mat, n_labels, opt_val = full_digits()

linear_objective = LinearObjective(C_mat, C_mat.shape)
sum_to_one_row_constraint = RowEqualityConstraint(C_mat.shape,
                                                  np.ones(C_mat.shape[1]), np.ones(C_mat.shape[1]), name='sum_to_one')
non_negativity_constraint = ElementWiseInequalityConstraint(C_mat.shape, 0,
                                                            name='nonnegativity')

alpha = n_labels
traceball = copt.constraint.TraceBall(alpha, C_mat.shape)
x_init = np.zeros(C_mat.shape).flatten()
beta0 = 1.

with open('stats.txt', 'a', buffering=1) as statsfile:
    cb = HomotopyTrace(stats_filehandle=statsfile)

    minimize_homotopy_cgm(
        linear_objective.f_grad,
        [sum_to_one_row_constraint, non_negativity_constraint],
        x_init,
        traceball.lmo,
        beta0,
        tol = 0,
        callback=cb,
        max_iter=int(1e5)
    )
