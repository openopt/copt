
from copt.prox_tv import prox_tv2d, prox_tv1d_rows, prox_tv1d_cols
from copt import davis_yin, proximal_gradient
from copt.utils import Trace
import numpy as np


def test_trace():
    n_samples = 100
    n_rows, n_cols = 5, 5
    n_features = n_rows * n_cols
    A = np.random.randn(n_samples, n_features)
    b = np.sign(np.random.randn(n_samples))

    def obj_fun(x):
        return 0.5 * np.linalg.norm(b - A.dot(x)) ** 2 / A.shape[0]

    def grad(x):
        return - A.T.dot(b - A.dot(x)) / A.shape[0]

    def no_prox(x, y): return x
    trace_three = Trace(lambda x: obj_fun(x))
    davis_yin(
        obj_fun, grad, no_prox, no_prox, np.zeros(n_features),
        callback=trace_three)
    # make sure that values are decreasing
    assert np.all(np.diff(trace_three.values) <= 0)

