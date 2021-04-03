import numpy as np
import pytest
from numpy import testing

import copt.constraint
import copt.penalty
from copt import tv_prox

proximal_penalties = [
    copt.penalty.L1Norm(1.0),
    copt.penalty.GroupL1(1.0, np.array_split(np.arange(16), 5)),
    copt.penalty.OGroupL1(1.0, np.array_split(np.arange(16), 5)),
    copt.penalty.TraceNorm(1.0, (4, 4)),
    copt.constraint.TraceBall(1.0, (4, 4)),
    copt.penalty.TotalVariation2D(1.0, (4, 4)),
    copt.penalty.FusedLasso(1.0),
]


def test_GroupL1():
    # non-overlapping
    groups = [(0, 1), (1, 2)]
    with np.testing.assert_raises(ValueError):
        copt.penalty.GroupL1(1.0, groups)

    # converts group type from tuple to list
    groups = [(0, 1), (2, 3)]
    pen = copt.penalty.GroupL1(1.0, groups)
    for g in pen.groups:
        np.testing.assert_(isinstance(g, list))

    # same number of groups and weights
    groups = [[0, 1], [2, 3]]
    weights = [1, 2, 3]
    with np.testing.assert_raises(ValueError):
        copt.penalty.GroupL1(1.0, groups, weights)

    # evaluation of penalty and prox
    x = np.array([0.01, 0.5, 3, 4])
    weights = np.array([10, .2])
    g0 = copt.penalty.GroupL1(1, groups, weights)
    # eval
    result = g0(x)
    gt = (weights[0] * np.linalg.norm(x[groups[0]], 2) +
          weights[1] * np.linalg.norm(x[groups[1]], 2))
    np.testing.assert_almost_equal(result, gt)
    # prox
    gt = x.copy()
    # the first group has norm lower than the corresponding weight
    gt[groups[0]] = 0
    # the second group has norm higher than the corresponding weight
    gt[groups[1]] -= (x[groups[1]] * weights[1] /
                      np.linalg.norm(x[groups[1]]))
    prox = g0.prox(x, 1)
    np.testing.assert_almost_equal(prox, gt)

    # default weights
    g1 = copt.penalty.GroupL1(1, groups)
    gt = np.array([1., 1])
    np.testing.assert_almost_equal(g1.weights, gt)

    # weights: sqrt(|g|)
    g2 = copt.penalty.GroupL1(1.0, groups, 'nf')
    gt = np.array([np.sqrt(2), np.sqrt(2)])
    np.testing.assert_almost_equal(g2.weights, gt)

    # weights: sqrt(|g|) ** -1
    g3 = copt.penalty.GroupL1(1.0, groups, 'nfi')
    gt = 1. / gt
    np.testing.assert_almost_equal(g3.weights, gt)

    # custom weights
    gt = np.random.rand(len(groups))
    g4 = copt.penalty.GroupL1(1.0, groups, gt)
    np.testing.assert_almost_equal(g4.weights, gt)
    expected = (np.linalg.norm(x[[0, 1]]) * gt[0] +
                np.linalg.norm(x[[2, 3]]) * gt[1])
    np.testing.assert_almost_equal(g4(x), expected)

    # sparse proximal
    _, B = g1.prox_factory(5)
    gt = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0],
        ]
    )
    np.testing.assert_almost_equal(B.toarray(), gt)

    groups = [(0, 1), (3, 4)]
    g5 = copt.penalty.GroupL1(1.0, groups)
    _, B = g5.prox_factory(5)
    gt = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, -1.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_almost_equal(B.toarray(), gt)


def test_OverlappingGroupL1():
    groups = [(0, 1), (2, 3)]
    g1 = copt.penalty.OGroupL1(1.0, groups)
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
    g2 = copt.penalty.OGroupL1(1.0, groups)
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
        x_next = tv_prox.prox_tv2d(x, gamma, n_rows, n_cols, tol=1e-10,
                                   max_iter=10000)
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
