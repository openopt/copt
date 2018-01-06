"""
Effect of acceleration in gradient descent
==========================================

Showcase the improved convergence of accelerated gradient
descent on a logistic regression problem.
"""
import numpy as np
import pylab as plt
import copt as cp

# .. construct (random) dataset ..
n_samples, n_features = 1000, 200
np.random.seed(0)
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))

logloss = cp.utils.logloss(X, y)
cb_pgd = cp.utils.Trace()
cb_apgd = cp.utils.Trace()
result_pgd = cp.minimize_PGD(
    logloss, np.zeros(n_features), callback=cb_pgd)
result_apgd = cp.minimize_APGD(
    logloss, np.zeros(n_features), callback=cb_apgd)

trace_func_pgd = np.array([logloss(x)[0] for x in cb_pgd.trace_x])
trace_func_apgd = np.array([logloss(x)[0] for x in cb_apgd.trace_x])
fmin = min(np.min(trace_func_apgd), np.min(trace_func_pgd))
plt.title('Comparison of full gradient optimizers')
plt.plot(trace_func_pgd - fmin, lw=4,
         label='gradient descent')
plt.plot(trace_func_apgd - fmin, lw=4,
         label='accelerated gradient descent')
plt.ylabel('Function suboptimality', fontweight='bold')
plt.xlabel('gradient evaluations', fontweight='bold')
plt.yscale('log')
plt.ylim(ymin=1e-10)
plt.xlim((0, 150))
plt.legend()
plt.grid()
plt.show()
