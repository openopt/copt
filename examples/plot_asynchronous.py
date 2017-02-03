"""
Asynchronous Stochastic Gradient
================================

"""
import numpy as np
from scipy import linalg
import pylab as plt
colors = ['#7fc97f', '#beaed4', '#fdc086']


from sklearn import datasets
from copt import stochastic
from scipy import sparse

# generate a random large sparse matrix as input data
# and associated target labels
n_samples, n_features = 50000, 10000
X = sparse.random(n_samples, n_features, density=0.001, format='csr')
w = sparse.random(1, n_features, density=0.01).toarray().ravel()
y = np.sign(X.dot(w) + np.random.randn(n_samples))


alpha = 1.0 / n_samples
beta = 1.0 / n_samples
step_size = stochastic.compute_step_size('logistic', X, alpha)
opt = stochastic.fmin_SAGA(
    stochastic.f_logistic, stochastic.deriv_logistic, X, y, np.zeros(X.shape[1]),
    step_size=step_size, alpha=alpha, beta=beta, max_iter=20,
    trace=True, verbose=True, g_prox=stochastic.prox_L1, g_func=stochastic.f_L1)


opt_4cores = stochastic.fmin_SAGA(
    stochastic.f_logistic, stochastic.deriv_logistic, X, y, np.zeros(X.shape[1]), step_size=step_size,
    alpha=alpha, beta=beta, tol=0, max_iter=20,
    trace=True, verbose=True, g_prox=stochastic.prox_L1, g_func=stochastic.f_L1, n_jobs=4)

fmin = min(np.min(opt.trace_func), np.min(opt_4cores.trace_func))
plt.plot(opt.trace_time, opt.trace_func - fmin, lw=4, label='1 core')
plt.plot(opt_4cores.trace_time, opt_4cores.trace_func - fmin, lw=4, label='4 cores')
plt.yscale('log')
plt.ylabel('Function suboptimality')
plt.xlabel('Time')
plt.grid()
plt.legend()
plt.show()