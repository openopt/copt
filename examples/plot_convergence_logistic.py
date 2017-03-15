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
X, y = datasets.load_rcv1()

logloss = loss.LogisticLoss(X, y)
result_pgd = fmin_PGD(logloss, trace=True)
result_apgd = fmin_APGD(logloss, trace=True)

fmin = np.min(result_pgd.trace_func)
plt.title('Logistic regression on RCV1 dataset')
plt.plot(result_pgd.trace_func - fmin, lw=4,
         label='gradient descent')
plt.plot(result_apgd.trace_func - fmin, lw=4,
         label='accelerated gradient descent')
plt.ylabel('Function suboptimality', fontweight='bold')
plt.xlabel('gradient evaluations', fontweight='bold')
plt.yscale('log')
plt.ylim(ymin=1e-3)
plt.xlim(xmax=50)
plt.legend()
plt.grid()
plt.show()
