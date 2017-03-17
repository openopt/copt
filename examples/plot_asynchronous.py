"""
Asynchronous Stochastic Gradient
================================

Comparison of asynchronous SAGA (ASAGA) for different number of cores.
"""
import numpy as np
import pylab as plt
import copt as cp

X, y = cp.datasets.load_rcv1()

f = cp.LogisticLoss(X, y, alpha=1e3)
g = cp.L1Norm(1e-5)

# opt_1cores = cp.minimize_SAGA(
#     f, max_iter=100, trace=True)
#
# opt_2cores = cp.minimize_SAGA(
#     f, max_iter=100, trace=True, n_jobs=2)

opt3 = cp.minimize_PGD(f, max_iter=100, trace=True)

# .. plot result ..
fmin = np.min(opt3.trace_func)
# plt.plot(opt_1cores.trace_func - fmin, lw=4, label='1 core')
# plt.plot(opt_2cores.trace_func - fmin, lw=4, label='2 cores')
plt.plot(opt3.trace_func - fmin, lw=4, label='PGD')
plt.yscale('log')
plt.ylabel('Function suboptimality')
plt.xlabel('Time (seconds)')
# plt.xlim(xmax=100)
# plt.ylim(ymin=1e-10)
plt.grid()
plt.legend()
plt.show()