import scipy
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
import numpy as np
import numpy.matlib as matlib
import scipy.linalg as linalg
import scipy.sparse.linalg as slinalg
import json

import copt.constraint

# mat = sio.loadmat("/Users/gideon/projects/SHCGM/Clustering/kmeans_sdp/data/data_features.mat")
mat = sio.loadmat("/Users/gideon/projects/SHCGM/Clustering/mydata.mat")
digits = mat['digits'].astype(np.float)
labels = mat['labels']
optval = 77.206632951040206
# k = max(np.argmax(mat['labels'],axis=0))+1
k = 10
D_mat = np.square(squareform(pdist(digits.T)))

dim,dim = D_mat.shape

# b = np.ones(dim).reshape(1,dim)
# b2 = np.ones(dim).reshape(dim,1)
# A = lambda x: np.sum(x,axis=1).reshape(1,dim)
# A2 = lambda x: np.sum(x,axis=0).reshape(dim,1)
# At = lambda y: matlib.repmat(y,dim,1).T
# At2 = lambda y: matlib.repmat(y,1,dim).T

x0 = np.zeros((dim,dim)).flatten()

alpha = k
traceball = copt.constraint.TraceBall(alpha, (dim,dim))
lmo = traceball.lmo

def smoothed_constraints_gradient(x, operator, offset):
    X = x.reshape((dim,dim))
    v = operator
    w = offset
    t_0 = X.dot(v) - w
    val = np.linalg.norm(t_0) ** 2
    grad = 2*np.outer(t_0, v)
    return val, grad

x = x0.copy()
n_iter = int(1e6)
beta0 = 1.
stats = []
ut, vt = None,None

normb = np.linalg.norm(np.ones(dim))

for it in range(n_iter):
    step_size = 2 / (it+2)
    beta_k = beta0/np.sqrt(it+2)

    X = x.reshape((dim,dim))
    # grad = beta_k*D_mat + At(A(X)-b) + At2(A2(X)-b2) + 1000*np.minimum(X,0)
    # g1 = At(A(X)-b) + At2(A2(X)-b2)
    _,g2 = smoothed_constraints_gradient(x, np.ones(dim), np.ones(dim))
    grad = beta_k*D_mat + g2 + 1000*np.minimum(X,0)
    grad = .5 * (grad+grad.T)
    # grad = grad.flatten()

    if False:
        # custom LMO
        # ut, _, vt = slinalg.svds(-grad, k=1, tol=1e-9, v0=ut)
        _,ut = slinalg.eigs(-grad, k=1, tol=1e-9, v0=ut, which='LR')
        ut = ut.real
        # vertex = k*np.outer(ut,vt)
        vertex = k*np.outer(ut,ut)
        X = (1-step_size)*X + step_size*vertex
        x = X.flatten()
    else:
        # copt lmo
        update_direction,_,_,_ = lmo(-grad, x, None)
        x += step_size*update_direction

    objective = np.dot(D_mat.flatten(), x.flatten())
    objective = np.abs(objective-optval) / np.abs(optval)
    # feasibility1 = np.linalg.norm(A(x.reshape((dim,dim)))-b) / np.linalg.norm(b)
    feasibility1 = np.linalg.norm(
        x.reshape((dim,dim)).dot(np.ones(dim)) - np.ones(dim)
    ) / normb
    feasibility2 = np.linalg.norm(np.minimum(x.reshape((dim,dim)),0), 'fro')

    stat = dict(iter=it, objective=objective, feasibility1=feasibility1, feasibility2=feasibility2)
    stats.append(stat)

    with open("stats.txt", "a+") as statsf:
        print(json.dumps(stat), file=statsf)
    
    if it % 100 == 0:
        print(stat)
        # print('x', x[0,0])


if True:
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