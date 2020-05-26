"""
Comparison of variants of Stochastic FW
===========================================

The problem solved in this case is a L1 constrained logistic regression
(sometimes referred to as sparse logistic regression).
"""

import copt as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# .. Load dataset ..
np.random.seed(0)
X, y = cp.datasets.load_rcv1("train")
dataset_name = "RCV1"
n_samples, n_features = X.shape
batch_size = 100
max_iter = int(1e1)
freq = max(max_iter * n_samples // (batch_size * 1000), 1)

# .. objective function and regularizer ..
f = cp.utils.LogLoss(X, y)
constraint = cp.utils.L1Ball(2e3)

# .. callbacks to track progress ..
def fw_gap(x):
    _, grad = f.f_grad(x)
    return constraint.lmo(-grad, x)[0].dot(-grad)


class TraceGaps(cp.utils.Trace):
    def __init__(self, f=None, freq=1):
        super(TraceGaps, self).__init__(f, freq)
        self.trace_gaps = []

    def __call__(self, dl):
        if self._counter % self.freq == 0:
            self.trace_gaps.append(fw_gap(dl['x']))
        super(TraceGaps, self).__call__(dl)


cb_sfw_SAG = TraceGaps(f, freq=freq)
cb_sfw_mokhtari = TraceGaps(f, freq=freq)
cb_sfw_lu_freund = TraceGaps(f, freq=freq)


# .. run the SFW algorithm ..
print("Running SAGFW")
result_sfw_SAG = cp.minimize_sfw(
    f.partial_deriv,
    X,
    y,
    np.zeros(n_features),
    constraint.lmo,
    batch_size,
    callback=cb_sfw_SAG,
    tol=0,
    max_iter=max_iter,
    variant='SAG'
)

print("Running MHK")
result_sfw_mokhtari = cp.minimize_sfw(
    f.partial_deriv,
    X,
    y,
    np.zeros(n_features),
    constraint.lmo,
    batch_size,
    callback=cb_sfw_mokhtari,
    tol=0,
    max_iter=max_iter,
    variant='MHK'
)

print("Running LF")
result_sfw_lu_freund = cp.minimize_sfw(
    f.partial_deriv,
    X,
    y,
    np.zeros(n_features),
    constraint.lmo,
    batch_size,
    callback=cb_sfw_lu_freund,
    tol=0,
    max_iter=max_iter,
    variant='LF'
)

print("Plotting...")
# .. plot the result ..
max_gap = max(cb_sfw_SAG.trace_gaps[0],
              cb_sfw_mokhtari.trace_gaps[0],
              cb_sfw_lu_freund.trace_gaps[0],
              )

max_val = max(np.max(cb_sfw_SAG.trace_fx),
              np.max(cb_sfw_mokhtari.trace_fx),
              np.max(cb_sfw_lu_freund.trace_fx),
              )

min_val = min(np.min(cb_sfw_SAG.trace_fx),
              np.min(cb_sfw_mokhtari.trace_fx),
              np.min(cb_sfw_lu_freund.trace_fx),
              )

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle("Sparse Logistic Regression -- {}".format(dataset_name), fontweight="bold")

ax1.plot(batch_size * freq * np.arange(len(cb_sfw_SAG.trace_gaps)), np.array(cb_sfw_SAG.trace_gaps) / max_gap, label="SFW -- SAG")
ax1.plot(batch_size * freq * np.arange(len(cb_sfw_mokhtari.trace_gaps)), np.array(cb_sfw_mokhtari.trace_gaps) / max_gap, label='SFW -- Mokhtari et al. (2020)')
ax1.plot(batch_size * freq * np.arange(len(cb_sfw_lu_freund.trace_gaps)), np.array(cb_sfw_lu_freund.trace_gaps) / max_gap, label='SFW -- Lu and Freund (2020)')
ax1.set_ylabel("Relative FW gap", fontweight="bold")
ax1.set_yscale('log')
ax1.grid()

matplotlib.rc('text', usetex=True)

ax2.plot(batch_size * freq * np.arange(len(cb_sfw_SAG.trace_fx)), (np.array(cb_sfw_SAG.trace_fx) - min_val) / (max_val - min_val), label="SFW -- Ours")
ax2.plot(batch_size * freq * np.arange(len(cb_sfw_mokhtari.trace_fx)), (np.array(cb_sfw_mokhtari.trace_fx) - min_val) / (max_val - min_val), label='SFW -- Mokhtari et al. (2018)')
ax2.plot(batch_size * freq * np.arange(len(cb_sfw_lu_freund.trace_fx)), (np.array(cb_sfw_lu_freund.trace_fx) - min_val) / (max_val - min_val), label='SFW -- Lu and Freund (2020)')
ax2.set_ylabel("Relative suboptimality", fontweight="bold")
ax2.set_xlabel("Number of gradient evaluations", fontweight="bold")
ax2.set_yscale("log")
ax2.legend()
ax2.grid()
plt.show()

print("Done.")