"""
Step-size strategies for FW
===========================

Plot showing how an optimal step-size for the Frank-Wolfe algorithm
evolves over time.
"""
import matplotlib.pylab as plt
from sklearn import datasets
from scipy import optimize
import numpy as np
import copt as cp

# Construct a toy classification dataset with 100 samples and 10 features
n_samples, n_features = 100, 10
X, y = datasets.make_classification(n_samples, n_features)


# Define an exact line search strategy
def exact_ls(kw):
    def f_ls(gamma):
        return kw['f_grad'](kw['x'] + gamma * kw['d_t'])[0]
    ls_sol = optimize.minimize_scalar(f_ls, bounds=[0, 1], method='bounded')
    return ls_sol.x


l1_ball = cp.utils.L1Ball(n_features / 2.)
f = cp.utils.LogLoss(X, y)
x0 = np.zeros(n_features)

cb = cp.utils.Trace(f=f)
out = cp.minimize_FW(
    f.f_grad, l1_ball.lmo, x0, callback=cb, max_iter=120,
    backtracking=exact_ls, L=f.lipschitz)

plt.plot(cb.trace_step_size)
plt.grid()
# start at iteration 10 since the first iterations tend to have a
# disproportionately large step-size
plt.xlim((10, 120))
plt.ylim(0, np.max(cb.trace_step_size[10:]))
plt.show()
