import numpy as np
from copt import fmin_PGD, fmin_APGD
from copt import tv_prox, loss

np.random.seed(0)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.sign(np.random.randn(n_samples))


def test_optimize():

    logloss = loss.LogisticLoss(X, y)
    opt = fmin_PGD(
        logloss, None, np.zeros(n_features))

    assert np.linalg.norm(logloss.gradient(opt.x)) < 0.01

    for alpha in np.logspace(-3, 3, 5):
        pen = loss.NormL1(alpha)
        opt = fmin_PGD(
            logloss, pen, np.zeros(n_features))
        ss = 1. / logloss.lipschitz_constant()
        gmap = (opt.x - pen.prox(opt.x - ss * logloss.gradient(opt.x), ss)) / ss
        assert np.linalg.norm(gmap) < 0.01
    #
    # opt = fmin_APGD(
    #     logloss, fprime_logloss, None, np.zeros(n_features),
    #     tol=1e-3)
    # sol_scipy = optimize.fmin_l_bfgs_b(
    #     logloss, np.zeros(n_features), fprime=fprime_logloss)[0]
    # assert opt.success
    # np.testing.assert_allclose(sol_scipy, opt.x, rtol=1e-1)



#
#
# def test_sklearn():
#     for alpha in np.logspace(-3, 3, 3):
#
#         def logloss(x):
#             return logistic._logistic_loss(x, X, y, 0.)
#
#         def fprime_logloss(x):
#             return logistic._logistic_loss_and_grad(x, X, y, 0.)[1]
#
#         clf = logistic.LogisticRegression(
#             penalty='l1', fit_intercept=False, C=1. / alpha)
#         clf.fit(X, y)
#         opt = fmin_PGD(
#             logloss, fprime_logloss, prox.prox_L1, np.zeros(n_features),
#             alpha=alpha, tol=1e-3, max_iter=1000)
#         assert opt.success
#         np.testing.assert_allclose(clf.coef_.ravel(), opt.x, atol=1e-3)
#
#         opt = fmin_APGD(
#             logloss, fprime_logloss, prox.prox_L1, np.zeros(n_features),
#             alpha=alpha, tol=1e-3)
#         assert opt.success
#         np.testing.assert_allclose(clf.coef_.ravel(), opt.x, atol=1e-3)
#
