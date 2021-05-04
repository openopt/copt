from copt.datasets import load_sdp_mnist
from copt.utils import Trace
from copt.loss import LinearLoss
from copt.homotopy import minimize_homotopy_cgm
import copt as cp
import copt.constraint
import copt.penalty
import numpy as np
import numpy.linalg as linalg
import pytest
from copt import tv_prox
from copt.constraint import (NonnegativeConstraint,
                             RowEqualityConstraint, euclidean_proj_simplex)
from numpy import testing
from scipy.sparse import linalg as splinalg

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


def test_homotopy():
    # TODO synthetic dataset here instead of a real one
    C_mat,n_labels,_ = load_sdp_mnist()

    linear_loss = LinearLoss(C_mat, C_mat.shape)
    sum_to_one_row_constraint = RowEqualityConstraint(C_mat.shape,
                                                    np.ones(C_mat.shape[1]), np.ones(C_mat.shape[1]), name='sum_to_one')
    non_negativity_constraint = NonnegativeConstraint(C_mat.shape, None, None, name='nonnegativity')

    spectrahedron = copt.constraint.TraceSpectrahedron(n_labels, C_mat.shape[0])

    beta0 = 1.
    x_init = np.zeros(C_mat.shape).flatten()

    cb = Trace(linear_loss)

    minimize_homotopy_cgm(
        linear_loss.f_grad,
        [sum_to_one_row_constraint, non_negativity_constraint],
        x_init,
        C_mat.shape,
        spectrahedron.lmo,
        beta0,
        tol = 0,
        callback=cb,
        max_iter=int(10)
    )

    assert np.isclose(cb.trace_fx[-1], 803.4766873414146)


