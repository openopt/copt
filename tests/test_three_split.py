import numpy as np
from copt.proximal_splitting import three_split
from sklearn.linear_model import logistic

n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():

    def logloss(x):
        return logistic._logistic_loss(x, X, y, 1.)

    def fprime_logloss(x):
        return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]

    opt = three_split(
        logloss, fprime_logloss, None, None,
        np.zeros(n_features), tol=1e-12)

    np.testing.assert_almost_equal(
        fprime_logloss(opt.x), np.zeros(n_features))


