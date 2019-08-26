"""
Group lasso with overlap
========================

Comparison of solvers for a least squares with
overlapping group lasso regularization.

References
----------
This example is modeled after the experiments in `Adaptive Three Operator Splitting <https://arxiv.org/pdf/1804.02339.pdf>`_, Appendix E.3.
"""
import copt as cp
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

np.random.seed(0)

n_samples, n_features = 100, 1002

# .. generate some data ..
# .. the first set of blocks is
groups = [np.arange(8 * i, 8 * i + 10) for i in range(125)]
ground_truth = np.zeros(n_features)
g = np.random.randint(0, len(groups), 10)
for i in g:
    ground_truth[groups[i]] = np.random.randn()

A = np.random.randn(n_samples, n_features)
p = 0.95  # create a matrix with correlations between features
for i in range(1, n_features):
    A[:, i] = p * A[:, i] + (1 - p) * A[:, i-1]
A[:, 0] /= np.sqrt(1 - p ** 2)
A = preprocessing.StandardScaler().fit_transform(A)
b = A.dot(ground_truth) + np.random.randn(n_samples)

# make labels in {0, 1}
b = np.sign(b)
b = (b + 1) // 2


# .. compute the step-size ..
max_iter = 5000
f = cp.utils.LogLoss(A, b)
step_size = 1. / f.lipschitz

# .. run the solver for different values ..
# .. of the regularization parameter beta ..
all_betas = np.logspace(-4, -1, 4)
all_trace_ls, all_trace_nols, all_trace_pdhg_nols, all_trace_pdhg = [], [], [], []
all_trace_ls_time, all_trace_nols_time, all_trace_pdhg_nols_time, all_trace_pdhg_time = [], [], [], []
out_img = []
for i, beta in enumerate(all_betas):
    print('beta = %s' % beta)
    G1 = cp.utils.GroupL1(beta, groups[::2])
    G2 = cp.utils.GroupL1(beta, groups[1::2])

    def loss(x):
        return f(x) + G1(x) + G2(x)

    cb_tosls = cp.utils.Trace()
    x0 = np.zeros(n_features)
    tos_ls = cp.minimize_three_split(
        f.f_grad, x0, G1.prox, G2.prox, step_size=10 * step_size,
        max_iter=max_iter, tol=1e-14, verbose=1,
        callback=cb_tosls, h_Lipschitz=beta)
    trace_ls = np.array([loss(x) for x in cb_tosls.trace_x])
    all_trace_ls.append(trace_ls)
    all_trace_ls_time.append(cb_tosls.trace_time)

    cb_tos = cp.utils.Trace()
    x0 = np.zeros(n_features)
    tos = cp.minimize_three_split(
        f.f_grad, x0, G1.prox, G2.prox,
        step_size=step_size,
        max_iter=max_iter, tol=1e-14, verbose=1,
        line_search=True, callback=cb_tos)
    trace_nols = np.array([loss(x) for x in cb_tos.trace_x])
    all_trace_nols.append(trace_nols)
    all_trace_nols_time.append(cb_tos.trace_time)
    out_img.append(tos.x)

    cb_pdhg = cp.utils.Trace()
    x0 = np.zeros(n_features)
    pdhg = cp.minimize_primal_dual(
        f.f_grad, x0, G1.prox, G2.prox,
        callback=cb_pdhg, max_iter=max_iter,
        step_size=step_size,
        step_size2=(1. / step_size) / 2, tol=0, line_search=False)
    trace_pdhg = np.array([loss(x) for x in cb_pdhg.trace_x])
    all_trace_pdhg.append(trace_pdhg)
    all_trace_pdhg_time.append(cb_pdhg.trace_time)

    cb_pdhg_nols = cp.utils.Trace()
    x0 = np.zeros(n_features)
    pdhg_nols = cp.minimize_primal_dual(
        f.f_grad, x0, G1.prox, G2.prox,
        callback=cb_pdhg_nols, max_iter=max_iter,
        step_size=step_size,
        step_size2=(1. / step_size) / 2, tol=0, line_search=False)
    trace_pdhg_nols = np.array([loss(x) for x in cb_pdhg_nols.trace_x])
    all_trace_pdhg_nols.append(trace_pdhg_nols)
    all_trace_pdhg_nols_time.append(cb_pdhg_nols.trace_time)


# .. plot the results ..
fig, ax = plt.subplots(2, 4, sharey=False)
xlim = [2000, 2000, 1000, 2000]
markevery = [x//5 for x in xlim]
for i, beta in enumerate(all_betas):
    ax[0, i].set_title(r'$\lambda=%s$' % beta)
    ax[0, i].set_title(r'$\lambda=%s$' % beta)
    ax[0, i].plot(out_img[i] / np.max(out_img[i]))
    ax[0, i].plot(ground_truth / np.max(ground_truth))
    ax[0, i].set_xticks(())
    ax[0, i].set_yticks(())
    ax[0, i].set_ylim((-0.5, 1.5))

    fmin = min(np.min(all_trace_ls[i]), np.min(all_trace_nols[i]))
    scale = 1. # all_trace_ls[i][0] - fmin
    plot_tos, = ax[1, i].plot(
        (all_trace_ls[i] - fmin) / scale, '--',
        lw=2, marker='o', markevery=markevery[i],
        markersize=5)

    plot_nols, = ax[1, i].plot(
        (all_trace_nols[i] - fmin) / scale,
        lw=2, marker='h', markevery=markevery[i],
        markersize=5)

    plot_pdhg, = ax[1, i].plot(
        (all_trace_pdhg[i] - fmin) / scale,
        lw=2, marker='^', markevery=markevery[i],
        markersize=5)

    plot_pdhg_nols, = ax[1, i].plot(
        (all_trace_pdhg_nols[i] - fmin) / scale,
        lw=2, marker='d', markevery=markevery[i],
        markersize=5)

    ax[1, i].set_xlabel('Iterations')
    ax[1, i].set_yscale('log')
    ax[1, i].set_ylim((1e-10, None))
    ax[1, i].set_xlim((0, xlim[i]))
    ax[1, i].grid(True)


plt.gcf().subplots_adjust(bottom=0.25)
plt.figlegend(
    (plot_tos, plot_nols, plot_pdhg, plot_pdhg_nols),
    ('TOS with line search', 'TOS without line search', 'PDHG with line search', 'PDHG without line search'), 'lower center', ncol=2,
    scatterpoints=1, frameon=False,)

ax[1, 0].set_ylabel('Objective minus optimum')
plt.show()
