# python3
"""
Total variation regularization
==============================

Comparison of solvers with total variation regularization.
"""
import copt as cp
import numpy as np
from PIL import Image
import pylab as plt
from scipy import misc
from scipy import sparse
from scipy.sparse import linalg as splinalg

np.random.seed(0)

img = misc.face(gray=True).astype(np.float)
# resize
img = np.array(Image.fromarray(img).resize((153, 115)))
img = img.astype(np.float) / img.max()

n_rows, n_cols = img.shape
n_features = n_rows * n_cols
n_samples = n_features
max_iter = 2000

# .. compute blurred and noisy image ..
A = sparse.load_npz('data/blur_matrix.npz')
b = A.dot(img.ravel()) + 0 * np.random.randn(n_samples)

np.random.seed(0)
n_samples = n_features

# .. compute the step-size ..
s = splinalg.svds(A, k=1, return_singular_vectors=False,
                  tol=1e-3, maxiter=500)[0]
f = cp.utils.SquareLoss(A, b)
step_size = 1./f.lipschitz


def loss(x, pen):
  x_mat = x.reshape((n_rows, n_cols))
  tmp1 = np.abs(np.diff(x_mat, axis=0))
  tmp2 = np.abs(np.diff(x_mat, axis=1))
  return f(x) + pen * (tmp1.sum() + tmp2.sum())


# .. run the solver for different values ..
# .. of the regularization parameter beta ..
all_betas = [0, 1e-6, 5e-6]
all_trace_ls, all_trace_nols, all_trace_pdhg, out_img = [], [], [], []
all_trace_ls_time, all_trace_nols_time, all_trace_pdhg_time = [], [], []
for i, beta in enumerate(all_betas):
  print('Iteration %s, beta %s' % (i, beta))

  def g_prox(x, gamma, pen=beta):
    return cp.tv_prox.prox_tv1d_cols(
        gamma * pen, x, n_rows, n_cols)

  def h_prox(x, gamma, pen=beta):
    return cp.tv_prox.prox_tv1d_rows(
        gamma * pen, x, n_rows, n_cols)

  cb_adatos = cp.utils.Trace()
  x0 = np.zeros(n_features)
  adatos = cp.minimize_three_split(
      f.f_grad, x0, g_prox, h_prox,
      step_size=10 * step_size,
      max_iter=max_iter, tol=1e-14, verbose=1,
      callback=cb_adatos, h_Lipschitz=beta)
  trace_ls = [loss(x, beta) for x in cb_adatos.trace_x]
  all_trace_ls.append(trace_ls)
  all_trace_ls_time.append(cb_adatos.trace_time)
  out_img.append(adatos.x.reshape(img.shape))

  cb_tos = cp.utils.Trace()
  x0 = np.zeros(n_features)
  cp.minimize_three_split(
      f.f_grad, x0, g_prox, h_prox,
      step_size=step_size,
      max_iter=max_iter, tol=1e-14, verbose=1,
      callback=cb_tos, line_search=False)
  trace_nols = [loss(x, beta) for x in cb_tos.trace_x]
  all_trace_nols.append(trace_nols)
  all_trace_nols_time.append(cb_tos.trace_time)

  cb_pdhg = cp.utils.Trace()
  x0 = np.zeros(n_features)
  cp.minimize_primal_dual(
      f.f_grad, x0, g_prox, h_prox,
      callback=cb_pdhg, max_iter=max_iter,
      step_size=step_size,
      step_size2=(1. / step_size) / 2, tol=0)
  trace_pdhg = np.array([loss(x, beta) for x in cb_pdhg.trace_x])
  all_trace_pdhg.append(trace_pdhg)
  all_trace_pdhg_time.append(cb_pdhg.trace_time)

# .. plot the results ..
f, ax = plt.subplots(2, 3, sharey=False)
xlim = [0.02, 0.02, 0.1]
for i, beta in enumerate(all_betas):
  ax[0, i].set_title(r'$\lambda=%s$' % beta)
  ax[0, i].imshow(out_img[i],
                  interpolation='nearest', cmap=plt.cm.gray)
  ax[0, i].set_xticks(())
  ax[0, i].set_yticks(())

  fmin = min(np.min(all_trace_ls[i]), np.min(all_trace_pdhg[i]))
  scale = all_trace_ls[i][0] - fmin
  plot_tos, = ax[1, i].plot(
      (all_trace_ls[i] - fmin) / scale, '--',
      lw=2, marker='o', markevery=400,
      markersize=7)

  plot_tos_nols, = ax[1, i].plot(
      (all_trace_nols[i] - fmin) / scale,
      lw=2, marker='<', markevery=400,
      markersize=7)

  plot_pdhg, = ax[1, i].plot(
      (all_trace_pdhg[i] - fmin) / scale, '--',
      lw=2, marker='^', markevery=400,
      markersize=7)

  ax[1, i].set_xlabel('Iterations')
  ax[1, i].set_yscale('log')
  ax[1, i].set_ylim((1e-14, None))
  ax[1, i].set_xlim((0, 1500))
  ax[1, i].grid(True)


plt.gcf().subplots_adjust(bottom=0.25)
plt.figlegend(
    (plot_tos, plot_tos_nols, plot_pdhg),
    ('Adaptive three operator splitting', 'three operator splitting', 'primal-dual hybrid gradient'), 'lower center', ncol=2,
    scatterpoints=1, frameon=False)

ax[1, 0].set_ylabel('Objective minus optimum')
plt.show()
