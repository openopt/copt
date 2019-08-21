"""Tests for the Frank-Wolfe algorithm."""
import copt as cp
import numpy as np
import pytest
from scipy import optimize

np.random.seed(0)
n_samples, n_features = 20, 16
A = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# we will use a logistic loss, which can't have values
# greater than 1
b = np.abs(b / np.max(np.abs(b)))

loss_funcs = [
    cp.utils.LogLoss,
    cp.utils.SquareLoss,
]


def test_fw_api():
  """Check that FW takes the right arguments and raises the right exceptions."""

  # test that the algorithm does not fail if x0
  # is a tuple
  f = cp.utils.LogLoss(A, b, 1. / n_samples)
  cb = cp.utils.Trace(f)
  alpha = 1.
  l1ball = cp.utils.L1Ball(alpha)
  cp.minimize_frank_wolfe(
      f.f_grad,
      [0]*n_features,
      l1ball.lmo,
      tol=0,
      lipschitz=f.lipschitz,
      callback=cb)

  # check that we riase an exception when the DR step-size is used
  # but no lipschitz constant is given
  with pytest.raises(ValueError):
    cp.minimize_frank_wolfe(
        f.f_grad,
        [0]*n_features,
        l1ball.lmo,
        step_size="DR")


@pytest.mark.parametrize("loss_grad", loss_funcs)
def test_fw_l1(loss_grad):
  """Test result of FW algorithm with L1 constraint."""
  f = loss_grad(A, b, 1. / n_samples)
  cb = cp.utils.Trace(f)
  alpha = 1.
  l1ball = cp.utils.L1Ball(alpha)
  opt = cp.minimize_frank_wolfe(
      f.f_grad,
      np.zeros(n_features),
      l1ball.lmo,
      tol=0,
      lipschitz=f.lipschitz,
      callback=cb)
  assert np.isfinite(opt.x).sum() == n_features

  ss = 1 / f.lipschitz
  grad = f.f_grad(opt.x)[1]
  grad_map = (opt.x - l1ball.prox(opt.x - ss * grad, ss)) / ss
  assert np.linalg.norm(grad_map) < 0.2


def exact_ls(kw):
  def f_ls(gamma):
    return kw["f_grad"](kw["x"] + gamma * kw["update_direction"])[0]
  ls_sol = optimize.minimize_scalar(f_ls, bounds=[0, 1], method="bounded")
  return ls_sol.x

@pytest.mark.parametrize("obj", loss_funcs)
@pytest.mark.parametrize("bt", [None, "adaptive", "adaptive2", "adaptive3", exact_ls])
def test_fw_backtrack(obj, bt):
  """Test FW with different options of the line-search strategy."""
  f = obj(A, b, 1. / n_samples)
  alpha = 2.
  traceball = cp.utils.TraceBall(alpha, (4, 4))
  opt = cp.minimize_frank_wolfe(
      f.f_grad,
      np.zeros(n_features),
      traceball.lmo,
      tol=0,
      lipschitz=f.lipschitz,
      step_size=bt)
  assert np.isfinite(opt.x).sum() == n_features

  ss = 1 / f.lipschitz
  grad = f.f_grad(opt.x)[1]
  grad_map = (opt.x - traceball.prox(opt.x - ss * grad, ss)) / ss
  assert np.linalg.norm(grad_map) < 1e-1


def exact_ls_pairwise(kw):
  def f_ls(gamma):
    return kw["f_grad"](kw["x"] + gamma * kw["update_direction"])[0]
  ls_sol = optimize.minimize_scalar(f_ls, bounds=[0, kw["max_step_size"]], method="bounded")
  assert ls_sol.x <= kw["max_step_size"]
  assert ls_sol.x >= 0
  return ls_sol.x


@pytest.mark.parametrize("obj", loss_funcs)
@pytest.mark.parametrize("step_size", ["DR", exact_ls_pairwise])
def test_pairwise_fw(obj, step_size):
  """Test the Pairwise FW method."""
  f = obj(A, b, 1. / n_samples)

  alpha = 1
  l1ball = cp.utils.L1Ball(alpha)
  x0 = np.zeros(A.shape[1])
  x0[0] = alpha
  active_set = np.zeros(2 * A.shape[1])
  active_set[0] = 1
  cb = cp.utils.Trace(f)
  opt = cp.minimize_pairwise_frank_wolfe(
      f.f_grad,
      x0,
      active_set,
      l1ball.lmo_pairwise,
      step_size=step_size,
      lipschitz=f.lipschitz,
      callback=cb)
  assert np.isfinite(opt.x).sum() == n_features

  ss = 1 / f.lipschitz
  grad = f.f_grad(opt.x)[1]
  grad_map = (opt.x - l1ball.prox(opt.x - ss * grad, ss)) / ss

  assert np.linalg.norm(grad_map) < 0.03
