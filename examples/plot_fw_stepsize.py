# python3
"""
Step-size and curvature on the Frank-Wolfe algorithm
====================================================

Plot showing both the optimal step-size and curvature for
the Frank-Wolfe algorithm on a logistic regression problem.

The step-size is computed as the one that gives the largest
decrease in objective function (see :func:`exact_ls`). The
curvature is computed as the largest eigenvalue of the
Hessian.

In the plot we can see how the variance of the step-size
is much higher than the one associated with the curvature.
"""
import copt as cp
import matplotlib.pylab as plt
import numpy as np
from scipy import optimize
from scipy.sparse import linalg as splinalg
from sklearn.preprocessing import robust_scale

# Construct a toy classification dataset with 100 samples and 10 features
# n_samples, n_features = 100, 10
# X, y = datasets.make_classification(n_samples, n_features, random_state=0)
datasets = [
    ("Gisette", cp.datasets.load_gisette),
    ("RCV1", cp.datasets.load_rcv1),
    ("Madelon", cp.datasets.load_madelon),
    ("Covtype", cp.datasets.load_covtype)]


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
for ax, (dataset_title, load_data) in zip(axes.ravel(), datasets):
  print("Running on the %s dataset" % dataset_title)

  X, y = load_data()
  n_samples, n_features = X.shape

  l1_ball = cp.utils.L1Ball(n_features / 2.)
  f = cp.utils.LogLoss(X, y)
  x0 = np.zeros(n_features)
  trace_step_size = []
  trace_curvature = []

  gamma0 = [0]

  # Define an exact line search strategy
  def exact_ls(kw):
    """Exact line-search for the Frank-Wolfe algorithm."""

    def f_ls(gamma):
      obj, grad_x = kw["f_grad"](kw["x"] + gamma[0] * kw["d_t"])
      grad_gamma = grad_x @ kw["d_t"]
      return obj, grad_gamma

    # Although there are scalar minimizations routines in scipy, the classical
    # L-BFGS-B method seems to be working much better at finding the optimal
    # step-size.
    ls_sol = optimize.minimize(
      f_ls, gamma0, bounds=[[0, 1]], method="L-BFGS-B", jac=True, tol=1e-20)

    # keep gamma0 as a warm start for next iterate
    gamma0[0] = ls_sol.x[0]
    trace_step_size.append(ls_sol.x)
    return ls_sol.x[0]


  def cb(kw):
    # trace_step_size.append(kw["step_size"])
    hessian = splinalg.LinearOperator(
        shape=(n_features, n_features), matvec=f.Hessian(kw["x"]))

    s, _ = splinalg.eigsh(hessian, k=1)
    trace_curvature.append(s)

  out = cp.minimize_frank_wolfe(
      f.f_grad,
      x0,
      l1_ball.lmo,
      callback=cb,
      max_iter=500,
      step_size=exact_ls,
      verbose=True
  )

  # Focus on the last 4/5, since the first iterations
  # tend to have a disproportionally large step-size
  n = len(trace_step_size) // 5
  trace_step_size = trace_step_size[n:]
  trace_curvature = trace_curvature[n:]

  color = "#67a9cf"
  ax.set_xlabel("number of iterations")
  ax.set_ylabel("step-size", color=color)
  ax.plot(
      n + np.arange(len(trace_step_size)),
      robust_scale(trace_step_size, with_centering=False),
      color=color,
      alpha=0.5)
  ax.tick_params(axis="y", labelcolor=color)

  ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

  color = "#ef8a62"
  ax2.set_ylabel(
      "curavature constant",
      color=color)  # we already handled the x-label with ax
  ax2.plot(
      n + np.arange(len(trace_curvature)), robust_scale(trace_curvature, with_centering=False), color=color, alpha=0.5)
  ax2.tick_params(axis="y", labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.xlim(n, n + len(trace_step_size))
plt.grid()
plt.show()
