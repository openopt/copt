import pytest
import numpy as np
from scipy import optimize
from copt.homotopy import RowEqualityConstraint

# use values which are arbitrary but reproducible
np.random.seed(0)

def test_row_equality_constraints_1d():
    one_dim = RowEqualityConstraint(1, np.ones(1), np.ones(1))
    assert one_dim.smoothed(np.array([1.]), 1.) == 0.
    assert one_dim.smoothed(np.array([0.]), 1.) == 0.5

    _, grad = one_dim.smoothed_g_grad(np.array([1.]),1.)
    assert grad == np.array([0.])

    _, grad = one_dim.smoothed_g_grad(np.array([0.]),1.)
    assert grad == np.array([-1.])

    _, grad = one_dim.smoothed_g_grad(np.array([2.]),1.)
    assert grad == np.array([1.])

    # check the functions that use beta
    for beta in [1e-5, 1e-3, 1e-1, 1, 10, 1e2]:
        for _ in range(2):
            f = lambda x: one_dim.smoothed_g_grad(x, beta)[0]
            g = lambda x: one_dim.smoothed_g_grad(x, beta)[1]
            x = np.random.randn(1)
            assert np.linalg.norm(optimize.check_grad(f, g, x)) < 1e-3

    # check the functions that don't handle beta
    for _ in range(20):
        f = lambda x: one_dim.grad_feasibility_dist_squared(x)[0]
        g = lambda x: one_dim.grad_feasibility_dist_squared(x)[1]
        x = np.random.randn(1)
        assert np.linalg.norm(optimize.check_grad(f, g, x)) < 1e-7


def test_row_equality_constraints_2d():
    # 2d case
    two_dim = RowEqualityConstraint((2,2), np.ones(2), np.ones(2))
    x = np.eye(2).flatten()
    assert x.shape == (4,)
    beta = 1.
    assert two_dim.smoothed(x, beta) == 0.
    assert two_dim.smoothed(np.ones(4)*.5, beta) == 0.
    assert two_dim.smoothed(np.ones(4), beta) == 1.

    # check the functions that use beta
    # N.B. precision is lost quickly as beta becomes tiny
    for beta in [1e-5, 1e-3, 1e-1, 1, 10, 1e2]:
        for _ in range(2):
            f = lambda x: two_dim.smoothed_g_grad(x, beta)[0]
            g = lambda x: two_dim.smoothed_g_grad(x, beta)[1]
            x = np.random.randn(4)
            assert np.linalg.norm(optimize.check_grad(f, g, x)) < 1e-1

    # check the functions that don't handle beta
    # N.B. some precision is lost as dimension increases from 1 to 2
    for _ in range(20):
        f = lambda x: two_dim.grad_feasibility_dist_squared(x)[0]
        g = lambda x: two_dim.grad_feasibility_dist_squared(x)[1]
        x = np.random.randn(4)
        assert np.linalg.norm(optimize.check_grad(f, g, x)) < 1e-6

@pytest.mark.parametrize("d", [10,20,50,100])
def test_row_equality_constraints_high_dim(d):
    eq_constr = RowEqualityConstraint((d,d), np.ones(d), np.ones(d))

    for _ in range(2):
        f = lambda x: eq_constr.grad(x)[0]
        g = lambda x: eq_constr.grad(x)[1]
        x = np.random.randn(d*d)
        assert np.linalg.norm(optimize.check_grad(f, g, x, epsilon=1e-6)) < 1e-3