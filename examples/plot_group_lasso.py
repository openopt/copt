"""
Group Lasso regularization
==========================

Comparison of solvers for problems with a group lasso regularization.

The group lasso regularization enters the optimization through
its proximal operator, which is implemented in copt through the
function prox of object :ref:`cp.utils.GroupL1`.

"""
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import pylab as plt
import copt as cp

np.random.seed(0)

# .. generate some data ..
n_samples, n_features = 100, 100
groups = [np.arange(10 * i, 10 * i + 10) for i in range(10)]

# .. construct a ground truth vector in which ..
# .. group 4 and 5 are nonzero ..
ground_truth = np.zeros(n_features)
ground_truth[groups[4]] = 1
ground_truth[groups[5]] = 0.5

max_iter = 5000
print('#features', n_features)

A = sparse.rand(n_samples, n_features, density=0.2)
sigma = 1.
b = A.dot(ground_truth) + sigma * np.random.randn(n_samples)

np.random.seed(0)
n_samples = n_features

# .. compute the step-size ..
s = splinalg.svds(A, k=1, return_singular_vectors=False,
                  tol=1e-3, maxiter=500)[0]
step_size = 1. / cp.utils.get_lipschitz(A, 'square')
f_grad = cp.utils.SquareLoss(A, b).func_grad

# .. run the solver for different values ..
# .. of the regularization parameter beta ..
all_betas = [0, 1e-2, 1e-1, 0.2]
all_trace_ls, all_trace_nols, all_trace_pdhg_nols, all_trace_pdhg = [], [], [], []
all_trace_ls_time, all_trace_nols_time, all_trace_pdhg_nols_time, all_trace_pdhg_time = [], [], [], []
out_img = []
for i, beta in enumerate(all_betas):
    print('beta = %s' % beta)
    G1 = cp.utils.GroupL1(beta, groups)

    def loss(x):
        return f_grad(x)[0] + G1(x)

    cb_tosls = cp.utils.Trace()
    x0 = np.zeros(n_features)
    cb_tosls(x0)
    pgd_ls = cp.minimize_PGD(
        f_grad, x0, G1.prox, step_size=step_size,
        max_iter=max_iter, tol=1e-14, verbose=1,
        callback=cb_tosls)
    trace_ls = np.array([loss(x) for x in cb_tosls.trace_x])
    all_trace_ls.append(trace_ls)
    all_trace_ls_time.append(cb_tosls.trace_time)

    cb_tos = cp.utils.Trace()
    x0 = np.zeros(n_features)
    cb_tos(x0)
    pgd = cp.minimize_PGD(
        f_grad, x0, G1.prox,
        step_size=step_size,
        max_iter=max_iter, tol=1e-14, verbose=1,
        line_search=False, callback=cb_tos)
    trace_nols = np.array([loss(x) for x in cb_tos.trace_x])
    all_trace_nols.append(trace_nols)
    all_trace_nols_time.append(cb_tos.trace_time)
    out_img.append(pgd.x)


# .. plot the results ..
fig, ax = plt.subplots(2, 4, sharey=False)
xlim = [0.02, 0.02, 0.1]
for i, beta in enumerate(all_betas):
    ax[0, i].set_title(r'$\lambda=%s$' % beta)
    ax[0, i].set_title(r'$\lambda=%s$' % beta)
    ax[0, i].plot(out_img[i])
    ax[0, i].plot(ground_truth)
    ax[0, i].set_ylim((-0.5, 1.5))
    ax[0, i].set_xticks(())
    ax[0, i].set_yticks(())

    fmin = min(np.min(all_trace_ls[i]), np.min(all_trace_nols[i]))
    scale = all_trace_ls[i][0] - fmin
    plot_tos, = ax[1, i].plot(
        (all_trace_ls[i] - fmin) / scale,
        lw=4, marker='o', markevery=100,
        markersize=10)

    plot_nols, = ax[1, i].plot(
        (all_trace_nols[i] - fmin) / scale,
        lw=4, marker='h', markevery=100,
        markersize=10)

    # plot_pdhg, = ax[1, i].plot(
    #     (all_trace_pdhg[i] - fmin) / scale,
    #     lw=4, marker='^', markevery=100,
    #     markersize=10)
    #
    # plot_pdhg_nols, = ax[1, i].plot(
    #     (all_trace_pdhg_nols[i] - fmin) / scale,
    #     lw=4, marker='d', markevery=100,
    #     markersize=10)

    ax[1, i].set_xlabel('Iterations')
    ax[1, i].set_yscale('log')
    ax[1, i].set_ylim((1e-14, None))
    ax[1, i].grid(True)


plt.gcf().subplots_adjust(bottom=0.15)
plt.figlegend(
    (plot_tos, plot_nols),
    ('PGD with line search', 'PGD without line search'), ncol=5,
    scatterpoints=1,
    loc=(-0.00, -0.0), frameon=False,
    bbox_to_anchor=[0.05, 0.01])

ax[1, 0].set_ylabel('Objective minus optimum')
plt.show()
