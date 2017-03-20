import numpy as np
import copt as cp

np.random.seed(0)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():

    for alpha in np.logspace(-3, 3, 5):
        for f in (cp.LogisticLoss(X, y, alpha), cp.SquaredLoss(X, y, alpha)):
            g = cp.L1Norm(alpha)
            for opt_alg in (cp.minimize_PGD, cp.minimize_APGD):
                opt = opt_alg(f, g)
                ss = 1. / f.lipschitz_constant()
                gmap = (opt.x - g.prox(opt.x - ss * f.gradient(opt.x), ss)) / ss
                assert np.linalg.norm(gmap) < 1e-6

