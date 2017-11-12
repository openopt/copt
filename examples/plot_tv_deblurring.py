"""
Total variation regularization
==============================

Comparison of solvers with total variation regularization.
"""
import numpy as np
from scipy import ndimage, misc, sparse
from scipy.sparse import linalg as splinalg
import pylab as plt
import copt as cp

np.random.seed(0)

img = misc.face(gray=True)
n_rows, n_cols = img.shape
n_features = n_rows * n_cols
n_samples = n_features # // 2
max_iter = 100
print('#features', n_features)


def blur_oper(x):
    x_img = x.reshape(img.shape)
    return ndimage.gaussian_filter(x_img, 10.)
A = splinalg.LinearOperator(
    matvec=blur_oper, rmatvec=blur_oper, dtype=np.float,
    shape=(n_features, n_features))

# A = sparse.rand(n_samples, n_features, density=1e-6)
# A = splinalg.aslinearoperator(A)
b = A.matvec(img.ravel()) + 10 * np.random.randn(n_samples)

np.random.seed(0)
n_samples = n_features


def TV(w):
    img = w.reshape((n_rows, n_cols))
    tmp1 = np.abs(np.diff(img, axis=0))
    tmp2 = np.abs(np.diff(img, axis=1))
    return tmp1.sum() + tmp2.sum()

f, ax = plt.subplots(2, 3, sharey=False)
all_alphas = [0, 1e-8, 1e-6]
xlim = [0.02, 0.02, 0.1]
s = splinalg.svds(A, k=1, return_singular_vectors=False,
                   tol=1e-6, maxiter=100)[0]
step_size = A.shape[0] / (s * s)  # .. 1/L

for i, alpha in enumerate(all_alphas):
    print(i, alpha)


    def g_prox(x, step_size):
        return cp.tv_prox.prox_tv1d_cols(
            step_size * alpha, x, n_rows, n_cols)


    def h_prox(x, step_size):
        return cp.tv_prox.prox_tv1d_rows(
            step_size * alpha, x, n_rows, n_cols)


    out_tos = cp.minimize_DavisYin(
        cp.utils.squareloss_grad(A, b),
        g_prox, h_prox, np.zeros(n_features),
        step_size=step_size,
        max_iter=max_iter, tol=1e-14, verbose=1, trace=True)

    tos_nols = cp.minimize_DavisYin(
        cp.utils.squareloss_grad(A, b),
        g_prox, h_prox, np.zeros(n_features),
        step_size=step_size,
        max_iter=max_iter, tol=1e-14, verbose=1, trace=True, backtracking=False)

#
#     trace_gd = Trace(lambda x: obj_fun(x) + alpha * TV(x))
#     f = cp.LogisticLoss(A, b, l2_reg)
#     g = cp.TotalVariation2D(alpha, n_rows, n_cols)
#     out_gd = cp.minimize_APGD(
#         f, g, max_iter=max_iter, callback=trace_gd)
#
    ax[0, i].set_title(r'$\lambda=%s$' % alpha)
    ax[0, i].imshow(out_tos.x.reshape((n_rows, n_cols)),
                    interpolation='nearest', cmap=plt.cm.gray)
    ax[0, i].set_xticks(())
    ax[0, i].set_yticks(())
#
    fmin = min(np.min(out_tos.trace_func), np.min(tos_nols.trace_func)) #, np.min(trace_gd.values))
    scale = (np.array(out_tos.trace_func) - fmin)[0]
    plot_tos, = ax[1, i].plot(
       np.array(out_tos.trace_time),
        (np.array(out_tos.trace_func) - fmin) / scale,
        lw=4, marker='o', markevery=10,
        markersize=10)

    plot_nols, = ax[1, i].plot(
       np.array(tos_nols.trace_time),
        (np.array(tos_nols.trace_func) - fmin) / scale,
        lw=4, marker='h', markevery=10,
        markersize=10)

#     prox_gd, = ax[1, i].plot(
#         np.array(trace_gd.times), (np.array(trace_gd.values) - fmin) / scale,
#         lw=4, marker='^', markersize=10, markevery=10,
#         color=colors[1])
    ax[1, i].set_xlabel('Time (in seconds)')
    ax[1, i].set_yscale('log')
    #ax[1, i].set_xlim((0, xlim[i]))
    ax[1, i].grid(True)

#
plt.gcf().subplots_adjust(bottom=0.15)
plt.figlegend(
    (plot_tos, plot_nols),
    ('TOS (LS)', 'TOS (no LS)'), ncol=5,
    scatterpoints=1,
    loc=(-0.00, -0.0), frameon=False,
    bbox_to_anchor=[0.05, 0.01])

ax[1, 0].set_ylabel('Objective minus optimum')
plt.show()
