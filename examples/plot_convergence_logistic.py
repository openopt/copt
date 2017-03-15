"""
Convergence of logistic regression
==================================

Implementation of logistic regression
using copt.
"""
import numpy as np
import pylab as plt
from copt import fmin_PGD, fmin_APGD
from copt import loss, datasets

# .. construct (random) dataset ..
n_samples, n_features = 1000, 1000
np.random.seed(0)
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))

logloss = loss.LogisticLoss(X, y)
result_pgd = fmin_PGD(logloss, trace=True)
result_apgd = fmin_APGD(logloss, trace=True, max_iter=200)

fmin = np.min(result_apgd.trace_func)
plt.title('Comparison of full gradient optimizers')
plt.plot(result_pgd.trace_func - fmin, lw=4,
         label='gradient descent')
plt.plot(result_apgd.trace_func - fmin, lw=4,
         label='accelerated gradient descent')
plt.ylabel('Function suboptimality', fontweight='bold')
plt.xlabel('gradient evaluations', fontweight='bold')
plt.yscale('log')
plt.ylim(ymin=1e-7)
plt.xlim((0, 100))
plt.legend()
plt.grid()
plt.show()
