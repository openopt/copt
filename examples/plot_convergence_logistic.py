"""
Convergence of logistic regression
==================================

Implementation of logistic regression
using copt.
"""
import numpy as np
import pylab as plt
from copt import fmin_PGD, loss
from copt import datasets

# .. construct (random) dataset ..
X, y = datasets.load_rcv1()

out = fmin_PGD(loss.LogisticLoss(X, y), trace=True)

plt.title('Logistic regression on RCV1 dataset')
plt.plot(out.trace_func - np.min(out.trace_func), lw=4,
         label='gradient descent')
plt.ylabel('Function suboptimality', fontweight='bold')
plt.xlabel('gradient evaluations', fontweight='bold')
plt.yscale('log')
plt.ylim(ymin=1e-15)
plt.xlim(xmax=25)
plt.legend()
plt.grid()
plt.show()
