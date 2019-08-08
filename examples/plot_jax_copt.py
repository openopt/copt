"""
Combining COPT with JAX
=======================

This example shows how `JAX <https://github.com/google/jax>`_
can be used within COPT to compute the gradients of the
objective function. In this example tensorflow-datasets
is used to provide the training data.
"""
import copt as cp
from jax import grad
from jax import numpy as np
import numpy as onp
import pylab as plt
from sklearn import datasets

# .. construct (random) dataset ..
X, y = datasets.make_regression()
n_samples, n_features = X.shape


def loss(w):
  # squared error loss
  z = np.dot(X, w) - y
  return np.sum(z * z) / n_samples


def f_grad(w):
  return loss(w), grad(loss)(w)

w0 = onp.zeros(n_features)

l1_ball = cp.utils.L1Norm(0.1)
cb = cp.utils.Trace(lambda x: loss(x) + l1_ball(x))
sol = cp.minimize_proximal_gradient(
    f_grad,
    w0,
    prox=l1_ball.prox,
    callback=cb
)
plt.plot(cb.trace_fx, lw=3)
plt.yscale('log')
plt.xlabel('# Iterations')
plt.ylabel('Objective value')
plt.grid()
plt.show()
