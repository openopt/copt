"""
Convergence of logistic regression
==================================

Implementation of logistic regression
using copt.
"""
import numpy as np
import pylab as plt
from copt import fmin_PGD, loss

# .. construct (random) dataset ..
n_samples, n_features = 5000, 1000
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

out = fmin_PGD(
    loss.LogisticLoss(X, y, 0), None, np.zeros(n_features),
    trace=True, tol=1e-12)

plt.plot(out.trace_time, out.trace_func - np.min(out.trace_func), lw=4)
plt.ylabel('Function suboptimality', fontweight='bold')
plt.xlabel('Time (in seconds)', fontweight='bold')
plt.yscale('log')
plt.ylim(ymin=1e-15)
plt.grid()
plt.show()
