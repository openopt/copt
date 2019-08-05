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
    return kw["f_grad"](kw["x"] + gamma * kw["d_t"])[0]
  ls_sol = optimize.minimize_scalar(f_ls, bounds=[0, 1], method="bounded")
  return ls_sol.x

@pytest.mark.parametrize("obj", loss_funcs)
@pytest.mark.parametrize("bt", [None, "adaptive", "adaptive2", "adaptive3", exact_ls])
def test_fw_backtrack(obj, bt):
  """Test FW with different options of the line-search strategy."""
  f = obj(A, b, 1. / n_samples)
  alpha = 1.
  traceball = cp.utils.TraceBall(alpha, (4, 4))
  opt = cp.minimize_frank_wolfe(
      f.f_grad,
      np.zeros(n_features),
      traceball.lmo,
      tol=0,
      # max_iter=5000,
      lipschitz=f.lipschitz,
      step_size=bt)
  assert np.isfinite(opt.x).sum() == n_features

  ss = 1 / f.lipschitz
  grad = f.f_grad(opt.x)[1]
  grad_map = (opt.x - traceball.prox(opt.x - ss * grad, ss)) / ss
  assert np.linalg.norm(grad_map) < 1e-1


# @pytest.mark.parametrize("obj", loss_funcs)
# @pytest.mark.parametrize("backtracking", [True, False])
# def test_pairwise_fw(obj, backtracking):
#   """Test the Pairwise FW method."""
#   f = obj(A, b, 1. / n_samples)

#   alpha = 1
#   l1ball = cp.utils.L1Ball(alpha)
#   cb = cp.utils.Trace(f)
#   opt = cp.minimize_pfw_l1(
#       f.f_grad,
#       alpha,
#       n_features,
#       tol=0,
#       max_iter=5000,
#       step_size=backtracking,
#       lipschitz=f.lipschitz,
#       callback=cb)
#   assert np.isfinite(opt.x).sum() == n_features

#   ss = 1 / f.lipschitz
#   grad = f.f_grad(opt.x)[1]
#   grad_map = (opt.x - l1ball.prox(opt.x - ss * grad, ss)) / ss

#   assert np.linalg.norm(grad_map) < 1e-10
