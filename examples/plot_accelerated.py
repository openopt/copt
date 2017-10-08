"""
Effect of acceleration in gradient descent
==========================================

Showcase the improved convergence of accelerated gradient
descent on a logistic regression problem.
"""
import numpy as np
import pylab as plt
from copt import utils, minimize_PGD, minimize_APGD

# .. construct (random) dataset ..
n_samples, n_features = 1000, 1000
np.random.seed(0)
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))

logloss = utils.LogisticLoss(X, y)
result_pgd = minimize_PGD(logloss, trace=True)
result_apgd = minimize_APGD(logloss, trace=True)

fmin = np.min(result_apgd.trace_func)
plt.title('Comparison of full gradient optimizers')
plt.plot(result_pgd.trace_func - fmin, lw=4,
         label='gradient descent')
plt.plot(result_apgd.trace_func - fmin, lw=4,
         label='accelerated gradient descent')
plt.ylabel('Function suboptimality', fontweight='bold')
plt.xlabel('gradient evaluations', fontweight='bold')
plt.yscale('log')
plt.ylim(ymin=1e-5)
plt.xlim((0, 300))
plt.legend()
plt.grid()
plt.show()
