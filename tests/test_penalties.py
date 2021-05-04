import copt as cp
import copt.constraint
import copt.penalty
import numpy as np
import numpy.linalg as linalg
import pytest
from copt import tv_prox
from copt.constraint import (NonnegativeConstraint,
                             RowEqualityConstraint)
from numpy import testing
from scipy.sparse import linalg as splinalg

proximal_penalties = [
    copt.penalty.L1Norm(1.0),
    copt.penalty.GroupL1(1.0, np.array_split(np.arange(16), 5)),
    copt.penalty.TraceNorm(1.0, (4, 4)),
    copt.constraint.TraceBall(1.0, (4, 4)),
    copt.constraint.TraceSpectrahedron(1.0, 4),
    copt.penalty.TotalVariation2D(1.0, (4, 4)),
    copt.penalty.FusedLasso(1.0),
]


def test_GroupL1():
    groups = [(0, 1), (2, 3)]
    g1 = copt.penalty.GroupL1(1.0, groups)
    _, B = g1.prox_factory(5)
    assert np.all(
        B.toarray()
        == np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0],
            ]
        )
    )

    groups = [(0, 1), (3, 4)]
    g2 = copt.penalty.GroupL1(1.0, groups)
    _, B = g2.prox_factory(5)
    assert np.all(
        B.toarray()
        == np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0],
            ]
        )
    )


#
#     for blocks in [[(0, 1), (2, 3)], ]:
#         pen = cp.utils.GroupL1(1., blocks)
#         counter = 0
#         for g in pen.groups:
#             for j in g:
#                 counter += 1
#         assert counter == blocks.size
#         assert pen.groups
#         for g in pen.groups:
#             assert np.unique(blocks[g]).size == 1


def test_tv1_prox():
    """
    Use the properties of strongly convex functions to test the implementation
    of the TV1D proximal operator. In particular, we use the following inequality
    applied to the proximal objective function: if f is mu-strongly convex then

          f(x) - f(x^*) >= ||x - x^*||^2 / (2 mu)

    where x^* is the optimum of f.
    """
    n_features = 10
    gamma = np.random.rand()
    epsilon = 1e-10  # account for some numerical errors

    tv_norm = lambda x: np.sum(np.abs(np.diff(x)))
    for _ in range(1000):
        x = np.random.randn(n_features)
        x_next = tv_prox.prox_tv1d(x, gamma)
        diff_obj = tv_norm(x) - tv_norm(x_next)
        testing.assert_array_less(
            ((x - x_next) ** 2).sum() / gamma, (1 + epsilon) * diff_obj
        )


def test_tv2_prox():
    """
    similar test, but for 2D total variation penalty.
    """
    np.random.seed(0)
    n_rows, n_cols = 6, 8
    n_features = n_rows * n_cols
    gamma = np.random.rand()
    epsilon = 0.1  # account for some numerical errors

    def tv_norm(x, n_rows, n_cols):
        X = x.reshape((n_rows, n_cols))
        return np.sum(np.abs(np.diff(X, 0))) + np.sum(np.abs(np.diff(X, 1)))

    for nrun in range(20):
        x = np.random.randn(n_features)
        x_next = tv_prox.prox_tv2d(x, gamma, n_rows, n_cols, tol=1e-10, max_iter=10000)
        diff_obj = tv_norm(x, n_rows, n_cols) - tv_norm(x_next, n_rows, n_cols)
        testing.assert_array_less(
            ((x - x_next) ** 2).sum() / gamma, (1 + epsilon) * diff_obj
        )


def test_tv2d_linear_operator():
    n_rows, n_cols = 20, 10

    def TV(w):
        img = w.reshape((n_rows, n_cols))
        tmp1 = np.abs(np.diff(img, axis=0))
        tmp2 = np.abs(np.diff(img, axis=1))
        return tmp1.sum() + tmp2.sum()

    L = tv_prox.tv2d_linear_operator(n_rows, n_cols)
    x = np.random.randn(n_rows * n_cols)
    testing.assert_almost_equal(np.abs(L.dot(x)).sum(), TV(x))


@pytest.mark.parametrize("pen", proximal_penalties)
def test_three_inequality(pen):
    """Test the L1 prox using the three point inequality

    The three-point inequality is described e.g., in Lemma 1.4
    in "Gradient-Based Algorithms with Applications to Signal
    Recovery Problems", Amir Beck and Marc Teboulle
    """
    n_features = 16

    for _ in range(10):
        z = np.random.randn(n_features)
        u = np.random.randn(n_features)
        xi = pen.prox(z, 1.0)

        lhs = 2 * (pen(xi) - pen(u))
        rhs = (
            np.linalg.norm(u - z) ** 2
            - np.linalg.norm(u - xi) ** 2
            - np.linalg.norm(xi - z) ** 2
        )
        assert lhs <= rhs, pen


# TODO is there a way of unifying the homotopy tests in the same elegant way as
# done above?

def test_elementwise_homotopy_constraint():
    n_features = 4
    shape = (n_features, n_features)

    for _ in range(10):
        operator = np.random.randn(*shape)
        offset = np.random.randn(1)[0]
        constraint = NonnegativeConstraint(shape, operator, offset, beta_scaling=1.)

        X = np.random.randn(*shape)
        _, grad = constraint.smoothed_grad(X)

        assert constraint(X-1000) == np.inf
        assert constraint(X+1000) == 0
        assert constraint(X-grad) == 0

def test_row_homotopy_lipschitz():
    n_features = 4
    shape = (n_features, n_features)

    for _ in range(100):
        operator = np.random.randn(n_features)
        offset = np.random.randn(n_features)
        constraint = RowEqualityConstraint(shape, operator, offset, beta_scaling=1.)

        z = np.random.randn(*shape)
        u = np.random.randn(*shape)
        _,zg = constraint.smoothed_grad(z)
        _,ug = constraint.smoothed_grad(u)

        # check Lipschitz continuous gradient
        L,_ = splinalg.eigsh(np.outer(operator, operator), k=1)
        lhs = np.linalg.norm(zg - ug)
        rhs = 2*L*np.linalg.norm(z - u)
        assert lhs <= rhs
