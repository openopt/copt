import numpy as np
from scipy import misc, linalg
from gdprox.prox_tv import prox_tv1d_cols, prox_tv1d_rows, prox_tv2d
from gdprox import fmin_three_split, fmin_prox_gd
from gdprox.utils import Trace
import pylab as plt

face = misc.imresize(misc.face(gray=True), 0.15)
face = face.astype(np.float) / 255.

# generate measurements as
# b = A ground_truth + noise
# where X is a random matrix
n_rows, n_cols = face.shape
n_features = face.shape[0] * face.shape[1]
np.random.seed(0)
n_samples = n_features // 100


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

alpha = 0.0
beta = 1.0
grad = lambda x: - A.T.dot(b - A.dot(x)) / A.shape[0] + alpha * x
f = lambda x: 0.5 * np.linalg.norm(b - A.dot(x)) ** 2 / A.shape[0] + 0.5 * alpha * x.dot(x)

# trace_three = Trace(lambda x: f(x) + beta * TV(x))
# fmin_three_split(f, grad, prox_tv1d_cols, prox_tv1d_rows, np.zeros(n_features), verbose=False, step_size=100,
#                  g_prox_args=(n_rows, n_cols), h_prox_args=(n_rows, n_cols), callback=trace_three,
#                  max_iter=500)

trace_gd = Trace(lambda x: f(x) + beta * TV(x))
fmin_prox_gd(f, grad, prox_tv2d, np.zeros(n_features), callback=trace_gd, g_prox_args=(n_rows, n_cols))

# fmin = np.min(trace_three.vals)
# plt.plot(trace_three.times, np.array(trace_three.vals) - fmin)
# plt.yscale('log')
# plt.grid()
# plt.show()