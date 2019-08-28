"""
Accelerated gradient descent
============================

Speed of convergence comparison between gradient descent
and Nesterov acceleration on a logistic regression problem.
"""
import copt as cp
import matplotlib.pyplot as plt
import numpy as np

# .. construct (random) dataset ..
n_samples, n_features = 1000, 200
np.random.seed(0)
X = np.random.randn(n_samples, n_features)
y = np.random.rand(n_samples)

f = cp.utils.LogLoss(X, y)
step_size = 1.0 / f.lipschitz

cb_pgd = cp.utils.Trace(f)
result_pgd = cp.minimize_proximal_gradient(
    f.f_grad,
    np.zeros(n_features),
    step_size=step_size,
    callback=cb_pgd,
    tol=0,
    accelerated=False,
)

cb_apgd = cp.utils.Trace(f)
result_apgd = cp.minimize_proximal_gradient(
    f.f_grad,
    np.zeros(n_features),
    step_size=step_size,
    callback=cb_apgd,
    tol=0,
    accelerated=True,
)


# .. plot the result ..
fmin = min(np.min(cb_pgd.trace_fx), np.min(cb_apgd.trace_fx))
plt.title("Comparison of full gradient optimizers")
plt.plot(cb_apgd.trace_fx - fmin, lw=4, label="accelerated gradient descent")
plt.plot(cb_pgd.trace_fx - fmin, lw=4, label="gradient descent")
plt.ylabel("Function suboptimality", fontweight="bold")
plt.xlabel("gradient evaluations", fontweight="bold")
plt.yscale("log")
plt.ylim(ymin=1e-16)
plt.xlim((0, 150))
plt.legend()
plt.grid()
plt.show()
