"""
Asynchronous Stochastic Gradient
================================

Comparison of asynchronous SAGA (ASAGA) for different number of cores.
"""
import numpy as np
import pylab as plt
import copt as cp

print('Loading data ...')
X, y = cp.datasets.load_rcv1()

f = cp.LogisticLoss(X, y)
g = cp.L1Norm(1e-6)

print('Running with 1 core ...')
opt_1cores = cp.minimize_SAGA(f, trace=True)

print('Running with 2 cores ...')
opt_2cores = cp.minimize_SAGA(f, trace=True, n_jobs=2)

# .. plot result ..
fmin = np.min(opt_1cores.trace_func)
plt.plot((opt_1cores.trace_func - fmin) / fmin, lw=4, marker='H', label='1 core')
plt.plot((opt_2cores.trace_func - fmin) / fmin, lw=4, marker='^', label='2 cores')
plt.yscale('log')
plt.ylabel('Function suboptimality')
plt.xlabel('Iterations (per core)')
plt.xlim(xmax=30)
# plt.ylim(ymin=1e-10)
plt.grid()
plt.legend()
plt.show()
