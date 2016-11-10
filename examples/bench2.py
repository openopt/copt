import numpy as np
from scipy import misc, linalg
from structsparse.prox_fast import prox_tv1d, prox_tv2d
from copt import fmin_three_split, proximal_gradient
from copt.utils import Trace
import pylab as plt
plt.style.use('dark_background')

face = misc.imresize(misc.face(gray=True), 0.15)
face = face.astype(np.float) / 255.

# generate measurements as
# b = A ground_truth + noise
# where X is a random matrix
n_rows, n_cols = face.shape
n_features = face.shape[0] * face.shape[1]
np.random.seed(0)
n_samples = n_features // 10


print('n_samples: %s, n_features: %s (%s)' % (n_samples, n_features, face.shape))
A = np.random.uniform(-1, 1, size=(n_samples, n_features))
for i in range(A.shape[0]):
    A[i] /= linalg.norm(A[i])
b = A.dot(face.ravel()) + 20.0 * np.random.randn(n_samples)
print(A.shape)


def TV(w):
    img = w.reshape((n_rows, n_cols))
    tmp1 = np.abs(np.diff(img, axis=0))
    tmp2 = np.abs(np.diff(img, axis=1))
    return tmp1.sum() + tmp2.sum()

l2_reg = 0
def obj_fun(x):
    return 0.5 * np.linalg.norm(b - A.dot(x)) ** 2 / A.shape[0] + 0.5 * l2_reg * x.dot(x)

def grad(x):
    return - A.T.dot(b - A.dot(x)) / A.shape[0] + l2_reg * x


def prox_tv1d_cols(a, stepsize, n_rows, n_cols):
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_cols):
        out[:, i] = prox_tv1d(A[:, i], stepsize)
    return out.ravel()


def prox_tv1d_rows(a, stepsize, n_rows, n_cols):
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_rows):
        out[i] = prox_tv1d(A[i, :], stepsize)
    return out.ravel()

from lightning.impl.sag import get_auto_step_size, get_dataset
ds = get_dataset(A, order="c")
eta = get_auto_step_size(ds, l2_reg, 'squared')

for beta in [0.01, 0.1, 1.0]:

    max_iter = 10000
    step_size = eta
    backtracking = False
    # from copt.prox_tv import prox_tv1d_cols, prox_tv1d_rows
    trace_three = Trace(lambda x: obj_fun(x) + beta * TV(x))
    fmin_three_split(obj_fun, grad,
                     prox_tv1d_cols,
                     prox_tv1d_rows,
                     np.zeros(n_features), verbose=False,
                     step_size=step_size, g_prox_args=(n_rows, n_cols), h_prox_args=(n_rows, n_cols),
                     callback=trace_three, max_iter=max_iter, tol=0., backtracking=backtracking)


    def prox_2d(x, stepsize, *args):
        return prox_tv2d(x.reshape(n_rows, n_cols), stepsize, *args).ravel()
    trace_gd = Trace(lambda x: obj_fun(x) + beta * TV(x))
    proximal_gradient(obj_fun, grad, prox_2d, np.zeros(n_features), callback=trace_gd, g_prox_args=(5, 1e-12),
                      step_size=step_size, max_iter=max_iter, tol=0., backtracking=backtracking)

    fmin = min(np.min(trace_three.vals), np.min(trace_gd.vals))
    scale = (np.array(trace_three.vals) - fmin)[0]
    plt.figure()
    plt.title(r'$\lambda=%s$' % beta)
    plt.plot(trace_three.times, (np.array(trace_three.vals) - fmin) / scale, label='TOS', lw=4, marker='o',
             markevery=500)
    plt.plot(trace_gd.times, (np.array(trace_gd.vals) - fmin) / scale, label='ProxGD', lw=4, marker='h',
             markevery=500)
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.savefig('bench2_%s.png' % beta)
    # plt.show()
