"""Tests for gradient-based methods
"""
import copt as cp
import numpy as np
import pytest
from scipy import optimize

np.random.seed(0)
n_samples, n_features = 20, 10
A = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# we will use a logistic loss, which can't have values
# greater than 1
b = np.abs(b / np.max(np.abs(b)))


loss_funcs = [cp.utils.LogLoss, cp.utils.SquareLoss, cp.utils.HuberLoss]
penalty_funcs = [None, cp.utils.L1Norm]


def test_gradient():
  for _ in range(20):
    A = np.random.randn(10, 5)
    b = np.random.rand(10)
    for loss in loss_funcs:
      f_grad = loss(A, b).f_grad
      f = lambda x: f_grad(x)[0]
      grad = lambda x: f_grad(x)[1]
      eps = optimize.check_grad(f, grad, np.random.randn(5))
      assert eps < 0.001


def certificate(x, grad_x, prox):
  if prox is None:
    def prox_(x, _):
      return x
  else:
    prox_ = prox

  return np.linalg.norm(x - prox_(x - grad_x, 1))


@pytest.mark.parametrize("accelerated", [True, False])
@pytest.mark.parametrize("loss", loss_funcs)
@pytest.mark.parametrize("penalty", penalty_funcs)
def test_optimize(accelerated, loss, penalty):
  """Test a method on both the line_search and fixed step size strategy."""
  max_iter = 2000
  for alpha in np.logspace(-1, 3, 3):
    obj = loss(A, b, alpha)
    if penalty is not None:
      prox = penalty(1e-3).prox
    else:
      prox = None
    opt = cp.minimize_proxgrad(
        obj.f_grad,
        np.zeros(n_features),
        prox=prox,
        tol=1e-12,
        step_size="adaptive",
        max_iter=max_iter,
        accelerated=accelerated)
    grad_x = obj.f_grad(opt.x)[1]
    assert certificate(opt.x, grad_x, prox) < 1e-6

    opt_2 = cp.minimize_proxgrad(
        obj.f_grad,
        np.zeros(n_features),
        prox=prox,
        max_iter=max_iter,
        tol=1e-12,
        step_size=1 / obj.lipschitz,
        accelerated=accelerated)
    grad_2x = obj.f_grad(opt_2.x)[1]
    assert certificate(opt_2.x, grad_2x, prox) < 1e-6


@pytest.mark.parametrize(
    "solver",
    [cp.minimize_proxgrad, cp.minimize_three_split, cp.minimize_primal_dual])
def test_callback(solver):
  """Make sure that the algorithm exists when the callback returns False."""

  def cb(_):
    return False

  f = cp.utils.SquareLoss(A, b)
  opt = solver(f.f_grad, np.zeros(n_features), callback=cb)
  assert opt.nit < 2


@pytest.mark.parametrize(
    "solver",
    [cp.minimize_proxgrad])
def test_line_search(solver):
  """Test the custom line search option."""

  def ls_wrong(_):
    return -10

  f = cp.utils.SquareLoss(A, b)
  opt = solver(f.f_grad, np.zeros(n_features), step_size=ls_wrong)
  assert not opt.success
