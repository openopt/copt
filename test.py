import numpy as np
from scipy import linalg, optimize
from sklearn.linear_model import logistic
from gdprox import fmin_prox_gd
from gdprox import prox_tv1d

n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():

    def logloss(x):
        return logistic._logistic_loss(x, X, y, 1.)

    def fprime_logloss(x):
        return logistic._logistic_loss_and_grad(x, X, y, 1.)[1]

    def g_prox(x, alpha):
        return x
    opt = fmin_prox_gd(
        logloss, fprime_logloss, g_prox, np.zeros(n_features),
        tol=1e-12, default_step_size=1)
    out = optimize.fmin_l_bfgs_b(
        logloss, np.zeros(n_features), fprime=fprime_logloss)
    assert linalg.norm(out[0] - opt.x) < 1e-3



def test_sklearn():
    for alpha in np.logspace(-3, 3):

        def logloss(x):
            return logistic._logistic_loss(x, X, y, 0.)

        def fprime_logloss(x):
            return logistic._logistic_loss_and_grad(x, X, y, 0.)[1]

        def g_prox(x, step_size):
            """
            L1 regularization
            """
            return np.fmax(x - step_size * alpha, 0) - \
                np.fmax(- x - step_size * alpha, 0)

        clf = logistic.LogisticRegression(
            penalty='l1', fit_intercept=False, C=1 / alpha)
        clf.fit(X, y)
        opt = fmin_prox_gd(
            logloss, fprime_logloss, g_prox, np.zeros(n_features),
            tol=1e-12)
        assert linalg.norm(opt.x - clf.coef_) < 1e-3

#
# def test_tv1_prox():
#     """
#     Use the properties of strongly convex functions to test the implementation
#     of the TV1D proximal operator. In particular, we use the following inequality
#     applied to the proximal objective function: if f is mu-strongly convex then
#
#           f(x) - f(x^*) >= ||x - x^*||^2 / (2 mu)
#
#     where x^* is the optimum of f.
#     """
#     n_features = 10
#     gamma = np.random.rand()
#
#     for nrun in range(100):
#         x = np.random.randn(n_features)
#         x2 = prox_tv1d(x, gamma)
#         diff_obj = np.sum(np.abs(x)) - np.sum(np.abs(x2))
#         assert diff_obj >= ((x - x2) ** 2).sum() / (2 * gamma)
#
#
# def test_tv2_prox():
#     """
#     similar test, but for 2D total variation penalty.
#     """
#     n_features = 36
#     gamma = np.random.rand()
#     pen = penalty.TotalVariation2DPenalty(6, 6)
#
#     for nrun in range(5):
#         x = np.random.randn(1, n_features)
#         x2 = pen.projection(x, gamma, 1)
#         diff_obj = pen.regularization(x) - pen.regularization(x2)
#         assert diff_obj >= ((x - x2) ** 2).sum() / (2 * gamma)