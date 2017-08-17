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
l2_reg = 1.0 / n_samples


A = np.random.uniform(-1, 1, size=(n_samples, n_features))
for i in range(A.shape[0]):
    A[i] /= linalg.norm(A[i])
b = A.dot(img.ravel()) + 1.0 * np.random.randn(n_samples)


def TV(w):
    img = w.reshape((n_rows, n_cols))
    tmp1 = np.abs(np.diff(img, axis=0))
    tmp2 = np.abs(np.diff(img, axis=1))
    return tmp1.sum() + tmp2.sum()


class TotalVariation1DCols:
    def __init__(self, alpha, n_rows, n_cols):
        self.alpha = alpha
        self.n_rows = n_rows
        self.n_cols = n_cols

    def __call__(self, x):
        img = x.reshape((self.n_rows, self.n_cols))
        tmp1 = np.abs(np.diff(img, axis=0))
        return self.alpha * tmp1.sum()

    def prox(self, x, step_size):
        return cp.tv_prox.prox_tv1d_cols(
            step_size * self.alpha, x, self.n_rows, self.n_cols)


class TotalVariation1DRows:
    def __init__(self, alpha, n_rows, n_cols):
        self.alpha = alpha
        self.n_rows = n_rows
        self.n_cols = n_cols

    def __call__(self, x):
        img = x.reshape((self.n_rows, self.n_cols))
        tmp2 = np.abs(np.diff(img, axis=1))
        return self.alpha * tmp2.sum()

    def prox(self, x, step_size):
        return cp.tv_prox.prox_tv1d_rows(
            step_size * self.alpha, x, self.n_rows, self.n_cols)


f, ax = plt.subplots(2, 3, sharey=False)
all_alphas = [1e-6, 1e-3, 1e-1]
xlim = [0.02, 0.02, 0.1]
for i, alpha in enumerate(all_alphas):
#
    max_iter = 5000
    out_tos = cp.minimize_DavisYin(
        cp.SquaredLoss(A, b, l2_reg), TotalVariation1DCols(alpha, n_rows, n_cols),
        TotalVariation1DRows(alpha, n_rows, n_cols), np.zeros(n_features),
        max_iter=max_iter, tol=1e-16, verbose=1)
#
#     trace_gd = Trace(lambda x: obj_fun(x) + alpha * TV(x))
#     f = cp.LogisticLoss(A, b, l2_reg)
#     g = cp.TotalVariation2D(alpha, n_rows, n_cols)
#     out_gd = cp.minimize_APGD(
#         f, g, max_iter=max_iter, callback=trace_gd)
#
#     ax[0, i].set_title(r'$\lambda=%s$' % alpha)
#     ax[0, i].imshow(out_tos.x.reshape((n_rows, n_cols)),
#                     interpolation='nearest', cmap=plt.cm.Blues)
#     ax[0, i].set_xticks(())
#     ax[0, i].set_yticks(())
#
#     fmin = min(np.min(trace_three.values), np.min(trace_gd.values))
#     scale = (np.array(trace_three.values) - fmin)[0]
#     prox_split, = ax[1, i].plot(
#         np.array(trace_three.times), (np.array(trace_three.values) - fmin) / scale,
#         lw=4, marker='o', markevery=10,
#         markersize=10, color=colors[0])
#     prox_gd, = ax[1, i].plot(
#         np.array(trace_gd.times), (np.array(trace_gd.values) - fmin) / scale,
#         lw=4, marker='^', markersize=10, markevery=10,
#         color=colors[1])
#     ax[1, i].set_xlabel('Time (in seconds)')
#     ax[1, i].set_yscale('log')
#     ax[1, i].set_xlim((0, xlim[i]))
#     ax[1, i].grid(True)
#
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.figlegend(
#     (prox_split, prox_gd),
#     ('Three operator splitting', 'Proximal Gradient Descent'), ncol=5,
#     scatterpoints=1,
#     loc=(-0.00, -0.0), frameon=False,
#     bbox_to_anchor=[0.05, 0.01])
#
# ax[1, 0].set_ylabel('Objective minus optimum')
# plt.show()
