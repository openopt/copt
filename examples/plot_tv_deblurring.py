"""
Total variation regularization
==============================

Comparison of solvers with total variation regularization.

TODO: split computational and plotting code
"""
import numpy as np
from scipy import misc
from scipy.sparse import linalg as splinalg
import pylab as plt
import copt as cp
from scipy.ndimage import gaussian_filter as gf

np.random.seed(0)

img = misc.face(gray=True)
n_rows, n_cols = img.shape
n_features = n_rows * n_cols
n_samples = n_features
max_iter = 1000
print('#features', n_features)


A = splinalg.LinearOperator(
    matvec=lambda x: gf(x.reshape(img.shape), 20.),
    rmatvec=lambda x: gf(x.reshape(img.shape), 20.),
    dtype=np.float, shape=(n_features, n_features))

b = A.matvec(img.ravel()) + 10 * np.random.randn(n_samples)

np.random.seed(0)
n_samples = n_features

# .. compute the step-size ..
s = splinalg.svds(A, k=1, return_singular_vectors=False,
                   tol=1e-2, maxiter=20)[0]
alpha = 0. / n_samples
L = cp.utils.lipschitz_constant(A, 'square', alpha)
step_size = 1. / L   # .. 1/L

f_grad = cp.utils.squareloss_grad(A, b, alpha)


def loss(x, beta):
    img = x.reshape((n_rows, n_cols))
    tmp1 = np.abs(np.diff(img, axis=0))
    tmp2 = np.abs(np.diff(img, axis=1))
    return f_grad(x)[0] + beta * (tmp1.sum() + tmp2.sum())


# .. run the solver for different values ..
# .. of the regularization parameter beta ..
all_betas = [0, 1e-6, 1e-4]
all_trace_ls, all_trace_nols, out_img = [], [], []
for i, beta in enumerate(all_betas):

    def g_prox(x, step_size):
        return cp.tv_prox.prox_tv1d_cols(
            step_size * beta, x, n_rows, n_cols)


    def h_prox(x, step_size):
        return cp.tv_prox.prox_tv1d_rows(
            step_size * beta, x, n_rows, n_cols)

    trace_x, trace_time = [], []

    def callback(kw):
        trace_x.append(kw['x'].copy())


    tos_ls = cp.minimize_DavisYin(
        f_grad, g_prox, h_prox, np.zeros(n_features),
        step_size=2 * step_size,
        max_iter=max_iter, tol=1e-14, verbose=1, trace=True,
        callback=callback)
    trace_ls = np.array([loss(x, beta) for x in trace_x])
    all_trace_ls.append(trace_ls)
    out_img.append(tos_ls.x.reshape(img.shape))

    trace_x = []
    tos_nols = cp.minimize_DavisYin(
        f_grad, g_prox, h_prox, np.zeros(n_features),
        step_size=step_size,
        max_iter=max_iter, tol=1e-14, verbose=1, trace=True,
        backtracking=False, callback=callback)
    trace_nols = np.array([loss(x, beta) for x in trace_x])
    all_trace_nols.append(trace_nols)


# .. plot the results ..
f, ax = plt.subplots(2, 3, sharey=False)
xlim = [0.02, 0.02, 0.1]
for i, beta in enumerate(all_betas):
    ax[0, i].set_title(r'$\lambda=%s$' % beta)
    ax[0, i].imshow(out_img[i],
                    interpolation='nearest', cmap=plt.cm.gray)
    ax[0, i].set_xticks(())
    ax[0, i].set_yticks(())

    fmin = min(np.min(all_trace_ls[i]), np.min(all_trace_nols[i]))
    scale = all_trace_ls[i][0] - fmin
    plot_tos, = ax[1, i].plot(
        (all_trace_ls[i] - fmin) / scale,
        lw=4, marker='o', markevery=10,
        markersize=10)

    plot_nols, = ax[1, i].plot(
        (all_trace_nols[i] - fmin) / scale,
        lw=4, marker='h', markevery=10,
        markersize=10)
    ax[1, i].set_xlabel('Time (in seconds)')
    ax[1, i].set_yscale('log')
    ax[1, i].grid(True)

plt.gcf().subplots_adjust(bottom=0.15)
plt.figlegend(
    (plot_tos, plot_nols),
    ('TOS (LS)', 'TOS (no LS)'), ncol=5,
    scatterpoints=1,
    loc=(-0.00, -0.0), frameon=False,
    bbox_to_anchor=[0.05, 0.01])

ax[1, 0].set_ylabel('Objective minus optimum')
plt.show()
