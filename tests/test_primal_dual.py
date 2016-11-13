import numpy as np
from copt.primal_dual import condat_vu

n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)