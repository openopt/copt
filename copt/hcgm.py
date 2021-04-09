import scipy
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
import numpy as np
import numpy.matlib as matlib
import scipy.linalg as linalg
import scipy.sparse.linalg as slinalg
import json

# mat = sio.loadmat("/Users/gideon/projects/SHCGM/Clustering/kmeans_sdp/data/data_features.mat")
mat = sio.loadmat("/Users/gideon/projects/SHCGM/Clustering/mydata.mat")
digits = mat['digits'].astype(np.float)
labels = mat['labels']
optval = 77.206632951040206
# k = max(np.argmax(mat['labels'],axis=0))+1
k = 10
D_mat = np.square(squareform(pdist(digits.T)))

dim,dim = D_mat.shape

b = np.ones(dim).reshape(1,dim)
b2 = np.ones(dim).reshape(dim,1)

A = lambda x: np.sum(x,axis=1).reshape(1,dim)
A2 = lambda x: np.sum(x,axis=0).reshape(dim,1)
At = lambda y: matlib.repmat(y,dim,1).T
At2 = lambda y: matlib.repmat(y,1,dim).T

x0 = np.zeros((dim,dim))

x = x0.copy()
n_iter = int(1e5)
beta0 = 1.
stats = []
for it in range(n_iter):
    step_size = 2 / (it+2)
    beta_k = beta0/np.sqrt(it+2)

    grad = beta_k*D_mat + At(A(x)-b) + At2(A2(x)-b2) + 1000*np.minimum(x,0)
    grad = .5 * (grad+grad.T)

    ut, _, vt = slinalg.svds(-grad, k=1, tol=1e-9)
    vertex = k*np.outer(ut,vt)

    x = (1-step_size)*x + step_size*vertex

    objective = np.dot(D_mat.flatten(), x.flatten())
    objective = np.abs(objective-optval) / np.abs(optval)
    feasibility1 = np.linalg.norm(A(x)-b) / np.linalg.norm(b)
    feasibility2 = np.linalg.norm(np.minimum(x,0), 'fro')

    stat = dict(iter=it, objective=objective, feasibility1=feasibility1, feasibility2=feasibility2)
    stats.append(stat)

    with open("stats.txt", "a+") as statsf:
        print(json.dumps(stat), file=statsf)
    
    if it % 100 == 0:
        print(stat)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([s['objective'] for s in stats])
ax.set_xscale('log')
ax.set_yscale('log')
fig.suptitle('objective')

fig, ax = plt.subplots()
ax.plot([s['feasibility1'] for s in stats])
ax.set_xscale('log')
ax.set_yscale('log')
fig.suptitle('feasibility 1')

fig, ax = plt.subplots()
ax.plot([s['feasibility2'] for s in stats])
ax.set_xscale('log')
ax.set_yscale('log')
fig.suptitle('feasibility 2')

plt.show()