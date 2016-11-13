import numpy as np
from copt.three_split import davis_yin
from sklearn.linear_model import logistic
from scipy import optimize, linalg

n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():

    def logloss(x):
        return logistic._logistic_loss(x, X, y, 1.)

    def fprime_logloss(x):
        return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]

    def g_prox(x, _):
        return x
    opt = davis_yin(
        logloss, fprime_logloss, g_prox, g_prox, np.zeros(n_features),
        tol=1e-12, step_size=1, verbose=True)
    1/0

    sol_scipy = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features), fprime=fprime_logloss)[0]
    np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-2)

    #
    # # same thing but using the other prox
    # opt = davies_yin(
    #     logloss, fprime_logloss, lambda x, y: x, g_prox, np.zeros(n_features),
    #     tol=1e-12)
    # assert linalg.norm(sol_scipy - opt.x) < 1e-3
