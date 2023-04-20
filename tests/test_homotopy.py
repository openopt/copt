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


