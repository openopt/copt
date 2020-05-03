"""
Stochastic FW
===========================================

The problem solved in this case is a L1 constrained logistic regression
(sometimes referred to as sparse logistic regression).
"""


import copt as cp
import matplotlib.pyplot as plt
import numpy as np

# .. construct (random) dataset ..
n_samples, n_features = 1000, 200
np.random.seed(0)
X = np.random.randn(n_samples, n_features)
y = np.random.rand(n_samples)

# .. objective function and regularizer ..
f = cp.utils.LogLoss(X, y)
constraint = cp.utils.L1Ball(1.)

# .. callbacks to track progress ..
cb_sfw_subopt = cp.utils.Trace(lambda x: f(x))

# .. run the SFW algorithm ..
result_sfw = cp.randomized.minimize_sfw(
    f.partial_deriv,
    X,
    y,
    np.zeros(n_features),
    constraint.lmo,
    callback=cb_sfw_subopt,
    tol=0,
    max_iter=int(1e5),
)

# .. plot the result ..
fmin = np.min(cb_sfw_subopt.trace_fx)
plt.title("Stochastic Frank-Wolfe")
plt.plot(cb_sfw_subopt.trace_fx - fmin, lw=4, label="SFW")
# .. for SVRG we multiply the number of iterations by two to ..
# .. account for computation of the snapshot gradient ..
plt.ylabel("Function suboptimality", fontweight="bold")
plt.xlabel("number of gradient evaluations", fontweight="bold")
plt.yscale("log")
plt.xlim((0, int(1e5)))
plt.legend()
plt.grid()
plt.show()
