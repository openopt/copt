import numpy as np
from scipy import sparse
from scipy import optimize
from datetime import datetime
from sklearn.utils.extmath import safe_sparse_dot

try:
    from numba import njit, prange
except ImportError:
    from functools import wraps

    def njit(*args, **kw):
        if len(args) == 1 and len(kw) == 0 and hasattr(args[0], "__call__"):
            func = args[0]

            @wraps(func)
            def inner_function(*args, **kwargs):
                return func(*args, **kwargs)

            return inner_function
        else:

            def inner_function(function):
                @wraps(function)
                def wrapper(*args, **kwargs):
                    return function(*args, **kwargs)

                return wrapper

            return inner_function
    
    prange = range


def build_func_grad(jac, fun, args, eps):
    if not callable(jac):
        if bool(jac):
            fun = optimize.optimize.MemoizeJac(fun)
            jac = fun.derivative
        elif jac == "2-point":
            jac = None
        else:
            raise NotImplementedError("jac has unexpected value.")

    if jac is None:

        def func_and_grad(x):
            f = fun(x, *args)
            g = optimize._approx_fprime_helper(x, fun, eps, args=args, f0=f)

    else:

        def func_and_grad(x):
            f = fun(x, *args)
            g = jac(x, *args)
            return f, g
    return func_and_grad


def safe_sparse_add(a, b):
    if sparse.issparse(a) and sparse.issparse(b):
        # both are sparse, keep the result sparse
        return a + b
    else:
        # one of them is non-sparse, convert
        # everything to dense.
        if sparse.issparse(a):
            a = a.toarray()
            if a.ndim == 2 and b.ndim == 1:
                b.ravel()
        elif sparse.issparse(b):
            b = b.toarray()
            if b.ndim == 2 and a.ndim == 1:
                b = b.ravel()
        return a + b


@njit(parallel=True)
def sample_batches(n_samples, n_batches, batch_size):
    idx = np.zeros(n_batches * batch_size, dtype=np.int32)
    for k in prange(n_batches):
        idx[k * batch_size:(k + 1) * batch_size] = np.random.choice(n_samples, size=batch_size, replace=False)
    return idx


@njit(nogil=True)
def fast_csr_vm(x, data, indptr, indices, d, idx):
    """
    Returns the vector matrix product x * M[idx]. M is described
    in the csr format.

    Returns x * M[idx]

    x: 1-d iterable
    data: data field of a scipy.sparse.csr_matrix
    indptr: indptr field of a scipy.sparse.csr_matrix
    indices: indices field of a scipy.sparse.csr_matrix
    d: output dimension
    idx: 1-d iterable: index of the sparse.csr_matrix
    """
    res = np.zeros(d)
    assert x.shape[0] == len(idx)
    for k, i in np.ndenumerate(idx):
        for j in range(indptr[i], indptr[i+1]):
            j_idx = indices[j]
            res[j_idx] += x[k] * data[j]
    return res


@njit(nogil=True)
def fast_csr_mv(data, indptr, indices, x, idx):
    """
    Returns the matrix vector product M[idx] * x. M is described
    in the csr format.

    data: data field of a scipy.sparse.csr_matrix
    indptr: indptr field of a scipy.sparse.csr_matrix
    indices: indices field of a scipy.sparse.csr_matrix
    x: 1-d iterable
    idx: 1-d iterable: index of the sparse.csr_matrix
    """

    res = np.zeros(len(idx))
    for i, row_idx in np.ndenumerate(idx):
        for k, j in enumerate(range(indptr[row_idx], indptr[row_idx+1])):
            j_idx = indices[j]
            res[i] += x[j_idx] * data[j]
    return res


def parse_step_size(step_size):
    if hasattr(step_size, "__len__") and len(step_size) == 2:
        return step_size[0], step_size[1]
    elif isinstance(step_size, float):
        return step_size, "fixed"
    elif hasattr(step_size, "__call__") or step_size == "adaptive":
        # without other information start with a step-size of one
        return 1, step_size
    else:
        raise ValueError("Could not understand value step_size=%s" % step_size)


class Trace:
    def __init__(self, f=None, freq=1):
        self.trace_x = []
        self.trace_time = []
        self.trace_fx = []
        self.trace_step_size = []
        self.start = datetime.now()
        self._counter = 0
        self.freq = int(freq)
        self.f = f

    def __call__(self, dl):
        if self._counter % self.freq == 0:
            if self.f is not None:
                self.trace_fx.append(self.f(dl["x"]))
            else:
                self.trace_x.append(dl["x"].copy())
            delta = (datetime.now() - self.start).total_seconds()
            self.trace_time.append(delta)
            self.trace_step_size.append(dl["step_size"])
        self._counter += 1


def init_lipschitz(f_grad, x0):
    L0 = 1e-3
    f0, grad0 = f_grad(x0)
    if sparse.issparse(grad0) and not sparse.issparse(x0):
        x0 = sparse.csc_matrix(x0).T
    elif sparse.issparse(x0) and not sparse.issparse(grad0):
        grad0 = sparse.csc_matrix(grad0).T
    x_tilde = x0 - (1.0 / L0) * grad0
    f_tilde = f_grad(x_tilde)[0]
    for _ in range(100):
        if f_tilde <= f0:
            break
        L0 *= 10
        x_tilde = x0 - (1.0 / L0) * grad0
        f_tilde = f_grad(x_tilde)[0]
    return L0


def get_max_lipschitz(A, loss, alpha=0):
    """
  XXX DEPRECATED

  Estimate the max Lipschitz constant (as appears in
  many stochastic methods).

  A : array-like

  loss : {"logloss", "square", "huber"}
  """
    from sklearn.utils.extmath import row_norms

    max_squared_sum = row_norms(A, squared=True).max()

    if loss == "logloss":
        return 0.25 * max_squared_sum + alpha
    elif loss in ("huber", "square"):
        raise NotImplementedError
    raise NotImplementedError


