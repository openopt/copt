"""
Total variation regularization
==============================

Comparison of solvers with total variation regularization.
"""
import numpy as np
from scipy import linalg
import pylab as plt
import copt as cp


###############################################################
# Load an ground truth image and generate the dataset (A, b) as
#
#             b = A ground_truth + noise   ,
#
# where A is a random matrix. We will now load the ground truth image
img = cp.datasets.load_img1()
n_rows, n_cols = img.shape
n_features = n_rows * n_cols
np.random.seed(0)
n_samples = n_features

# set L2 regularization (arbitrarily) to 1/n_samples


A = np.random.uniform(-1, 1, size=(n_samples, n_features))
for i in range(A.shape[0]):
    A[i] /= linalg.norm(A[i])
b = A.dot(img.ravel()) + 1.0 * np.random.randn(n_samples)

print(1)

def TV(w):
    img = w.reshape((n_rows, n_cols))
    tmp1 = np.abs(np.diff(img, axis=0))
    tmp2 = np.abs(np.diff(img, axis=1))
    return tmp1.sum() + tmp2.sum()




f, ax = plt.subplots(2, 3, sharey=False)
all_alphas = [1e-6, 1e-3, 1e-1]
xlim = [0.02, 0.02, 0.1]
for i, alpha in enumerate(all_alphas):
    print(i, alpha)


    def prox_g(x, step_size):
        return cp.tv_prox.prox_tv1d_cols(
            step_size * alpha, x, n_rows, n_cols)


    def prox_h(x, step_size):
        return cp.tv_prox.prox_tv1d_rows(
            step_size * alpha, x, n_rows, n_cols)

    max_iter = 50
    out_tos = cp.minimize_DavisYin(
        cp.utils.squareloss_grad(A, b), prox_g, prox_h, np.zeros(n_features),
        max_iter=max_iter, tol=1e-14, verbose=1, trace=True, step_size=1.0, backtracking=False)
    #
    # tos_nols = cp.minimize_DavisYin(
    #     cp.SquaredLoss(A, b), prox_g, prox_h, np.zeros(n_features),
    #     max_iter=max_iter, tol=1e-14, verbose=1, trace=True, backtracking=False)

#
#     trace_gd = Trace(lambda x: obj_fun(x) + alpha * TV(x))
#     f = cp.LogisticLoss(A, b, l2_reg)
#     g = cp.TotalVariation2D(alpha, n_rows, n_cols)
#     out_gd = cp.minimize_APGD(
#         f, g, max_iter=max_iter, callback=trace_gd)
#
    ax[0, i].set_title(r'$\lambda=%s$' % alpha)
    ax[0, i].imshow(out_tos.x.reshape((n_rows, n_cols)),
                    interpolation='nearest', cmap=plt.cm.Blues)
    ax[0, i].set_xticks(())
    ax[0, i].set_yticks(())
#
    fmin = np.min(out_tos.trace_func)
    scale = (np.array(out_tos.trace_func) - fmin)[0]
    plot_tos, = ax[1, i].plot(
       np.array(out_tos.trace_time),
        (np.array(out_tos.trace_func) - fmin) / scale,
        lw=4, marker='o', markevery=10,
        markersize=10)
    #
    # plot_nols, = ax[1, i].plot(
    #    np.array(tos_nols.trace_time),
    #     (np.array(tos_nols.trace_func) - fmin) / scale,
    #     lw=4, marker='h', markevery=10,
    #     markersize=10)

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
    (plot_tos,),
    ('TOS (LS)', ), ncol=5,
    scatterpoints=1,
    loc=(-0.00, -0.0), frameon=False,
    bbox_to_anchor=[0.05, 0.01])

ax[1, 0].set_ylabel('Objective minus optimum')
plt.show()
