"""
Total variation regularization
==============================

Comparison of solvers with total variation regularization.
"""
# import numpy as np
# from scipy import linalg
# import pylab as plt
# import copt as cp
# colors = ['#7fc97f', '#beaed4', '#fdc086']
#
# from datetime import datetime
#
# class Trace:
#     """
#     XXX
#     """
#     def __init__(self, loss_func, print_freq=100, verbose=False):
#         self.loss_func = loss_func
#         self.values = []
#         self.times = []
#         self.start = None
#         self.counter = 0
#         self.print_freq = print_freq
#         self.verbose = verbose
#
#     def __call__(self, args):
#         fxk = self.loss_func(args)
#         if self.verbose and self.counter % self.print_freq == 0:
#             print('Iteration: %s, Trace obj: %s' %
#                   (self.counter, fxk))
#         self.counter += 1
#         self.values.append(fxk)
#         if self.start is None:
#             self.start = datetime.now()
#             self.times = [0]
#         else:
#             self.times.append((datetime.now() - self.start).total_seconds())
#
#
# ###############################################################
# # Load an ground truth image and generate the dataset (A, b) as
# #
# #             b = A ground_truth + noise   ,
# #
# # where A is a random matrix. We will now load the ground truth image
# img = cp.datasets.load_img1()
# n_rows, n_cols = img.shape
# n_features = n_rows * n_cols
# np.random.seed(0)
# n_samples = n_features
#
# # set L2 regularization (arbitrarily) to 1/n_samples
# l2_reg = 1.0 / n_samples
#
#
# A = np.random.uniform(-1, 1, size=(n_samples, n_features))
# for i in range(A.shape[0]):
#     A[i] /= linalg.norm(A[i])
# b = A.dot(img.ravel()) + 1.0 * np.random.randn(n_samples)
#
#
# def TV(w):
#     img = w.reshape((n_rows, n_cols))
#     tmp1 = np.abs(np.diff(img, axis=0))
#     tmp2 = np.abs(np.diff(img, axis=1))
#     return tmp1.sum() + tmp2.sum()
#
#
# def obj_fun(x):
#     return 0.5 * np.linalg.norm(b - A.dot(x)) ** 2 / A.shape[0] + 0.5 * l2_reg * x.dot(x)
#
#
# def grad(x):
#     return - A.T.dot(b - A.dot(x)) / A.shape[0] + l2_reg * x
#
# f, ax = plt.subplots(2, 3, sharey=False)
# all_alphas = [1e-6, 1e-3, 1e-1]
# xlim = [0.02, 0.02, 0.1]
# for i, alpha in enumerate(all_alphas):
#
#     max_iter = 5000
#     trace_three = Trace(lambda x: obj_fun(x) + alpha * TV(x))
#     out_tos = cp.minimize_DavisYin(
#         obj_fun, grad, cp.tv_prox.prox_tv1d_rows, cp.tv_prox.prox_tv1d_cols, np.zeros(n_features),
#         alpha=alpha, beta=alpha, g_prox_args=(n_rows, n_cols), h_prox_args=(n_rows, n_cols),
#         callback=trace_three, max_iter=max_iter, tol=1e-16)
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
