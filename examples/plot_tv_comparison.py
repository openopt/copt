import numpy as np
from scipy import misc, linalg

face = misc.imresize(misc.face(gray=True), 0.15)
face = face.astype(np.float) / 255.

# generate measurements as
# b = A ground_truth + noise
# where X is a random matrix
n_rows, n_cols = face.shape
n_features = face.shape[0] * face.shape[1]
np.random.seed(0)



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


def loss(w):
    return 0.5 * (b - A.dot(w)) ** 2

beta = 1.0
gd_loss = []
gd_time = []
from datetime import datetime
start = datetime.now()

def gd_callback(w):

    full_loss = loss.mean() + beta * TV(w)
    global n_iter
    print(n_iter, full_loss)
    n_iter += 1
    gd_loss.append(full_loss)
    gd_time.append((datetime.now() - start).total_seconds())

from  gdprox import three_split

three_split