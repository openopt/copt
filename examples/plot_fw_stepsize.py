# python3
"""
Comparison of different step-sizes in Frank-Wolfe
=================================================


"""
import copt as cp
import matplotlib.pylab as plt
import numpy as np

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

  for step_size, label in [
      [(f.lipschitz, "adaptive"), "adaptive step-size"],
      [f.lipschitz, "Lipschitz step-size"]]:
    cb = cp.utils.Trace(f)
    cp.minimize_frank_wolfe(
        f.f_grad,
        x0,
        l1_ball.lmo,
        callback=cb,
        max_iter=20,
        step_size=step_size,
        verbose=True
    )
    ax.plot(cb.trace_fx, label=label)
  ax.set_xlabel("number of iterations")
  ax.set_ylabel("step-size")
  ax.set_title(dataset_title)
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  ax.grid()
plt.legend()
plt.show()
