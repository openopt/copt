"""
Asynchronous Stochastic Gradient
================================

"""
import numpy as np
import pylab as plt
from copt import stochastic
from scipy import sparse
colors = ['#7fc97f', '#beaed4', '#fdc086']

# generate a random large sparse matrix as input data
# and associated target labels
n_samples, n_features = 10000, 10000
X = sparse.random(n_samples, n_features, density=0.001, format='csr')
w = sparse.random(1, n_features, density=0.01).toarray().ravel()
y = np.sign(X.dot(w) + np.random.randn(n_samples))


alpha = 1.0 / n_samples
beta = 1.0 / n_samples
step_size = stochastic.compute_step_size('logistic', X, alpha)

max_iter = 25

opt_1cores = stochastic.fmin_SAGA(
    stochastic.f_logistic, stochastic.deriv_logistic, X, y, np.zeros(X.shape[1]),
    step_size=step_size, alpha=alpha, beta=beta, max_iter=max_iter, tol=-1,
    trace=True, verbose=True, g_prox=stochastic.prox_L1, g_func=stochastic.f_L1)


opt_2cores = stochastic.fmin_SAGA(
    stochastic.f_logistic, stochastic.deriv_logistic, X, y, np.zeros(X.shape[1]),
    step_size=step_size, alpha=alpha, beta=beta, max_iter=max_iter, tol=-1,
    trace=True, verbose=True, g_prox=stochastic.prox_L1, g_func=stochastic.f_L1, n_jobs=2)


opt_3cores = stochastic.fmin_SAGA(
    stochastic.f_logistic, stochastic.deriv_logistic, X, y, np.zeros(X.shape[1]),
    step_size=step_size, alpha=alpha, beta=beta, max_iter=max_iter, tol=-1,
    trace=True, verbose=True, g_prox=stochastic.prox_L1, g_func=stochastic.f_L1, n_jobs=3)

fmin = min(np.min(opt_1cores.trace_func), np.min(opt_2cores.trace_func),
           np.min(opt_3cores.trace_func))

plt.plot(opt_1cores.trace_time, opt_1cores.trace_func - fmin, lw=4, label='1 core',
         color=colors[0])
plt.plot(opt_2cores.trace_time, opt_2cores.trace_func - fmin, lw=4, label='2 cores',
         color=colors[1])
plt.plot(opt_3cores.trace_time, opt_3cores.trace_func - fmin, lw=4, label='3 cores',
         color=colors[2])

plt.yscale('log')
plt.ylabel('Function suboptimality')
plt.xlabel('Time')
plt.grid()
plt.legend()
plt.show()