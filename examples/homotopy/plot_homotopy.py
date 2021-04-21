import json
from pathlib import Path

import copt
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from copt.constraint import (ElementWiseInequalityConstraint,
                             RowEqualityConstraint)
from copt.homotopy import minimize_homotopy_cgm
from copt.loss import LinearLoss

from copt.datasets import load_sdp_mnist

class HomotopyTrace(copt.utils.Trace):
    """Trace callback for homotopy algorithms.

    Tracks the relative objective optimality as well as the approximate
    feasibility. This information is stored in `trace_relative_subopt` and
    `trace_feasibilities`. It is also appended to the file if one is
    specified.

    Args:
      stats_filehandle: File object
        The file where json dictionaries of the tracked information are
        written.
    """
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

C_mat, n_labels, opt_val = load_sdp_mnist(subset='reduced')

linear_objective = LinearLoss(C_mat, C_mat.shape)
sum_to_one_row_constraint = RowEqualityConstraint(C_mat.shape,
                                                  np.ones(C_mat.shape[1]), np.ones(C_mat.shape[1]), name='sum_to_one')
non_negativity_constraint = ElementWiseInequalityConstraint(C_mat.shape, 0,
                                                            name='nonnegativity')

# traceball = copt.constraint.TraceBall(n_labels, C_mat.shape)
spectrahedron = copt.constraint.TraceSpectrahedron(n_labels, C_mat.shape[0])

x_init = np.zeros(C_mat.shape).flatten()
beta0 = 1.

with open('stats.txt', 'a', buffering=1) as statsfile:
    cb = HomotopyTrace(stats_filehandle=statsfile)

    minimize_homotopy_cgm(
        linear_objective.f_grad,
        [sum_to_one_row_constraint, non_negativity_constraint],
        x_init,
        C_mat.shape,
        spectrahedron.lmo,
        beta0,
        tol = 0,
        callback=cb,
        max_iter=int(1e5)
    )

# plotting
with open('stats.txt', 'r') as f:
    stats = [json.loads(line) for line in f.readlines()]

fig, axs = plt.subplots(nrows=1,ncols=3,sharex=True,figsize=(21,7))

axs[0].loglog([s['objective'] for s in stats])
axs[1].loglog([s['sum_to_one'] for s in stats])
axs[2].loglog([s['nonnegativity'] for s in stats])

for ax,name in zip(axs,["objective", "sum_to_one", "nonnegativity"]):
    ax.set_title(name)

for ax in axs:
    ax.grid(True)

plt.show()

