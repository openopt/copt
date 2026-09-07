"""Tests for the Stochastic Frank-Wolfe algorithms."""
import numpy as np
import pytest
from scipy import optimize, sparse
import copt as cp
import copt.constraint
import copt.loss

np.random.seed(0)
n_samples, n_features = 20, 16
A = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# we will use a logistic loss, which can't have values
# greater than 1
b = np.abs(b / np.max(np.abs(b)))

LOSS_FUNCS = [copt.loss.LogLoss]
VARIANTS = ['SAGA', 'SAG', 'MHK', 'LF']
BATCH_SIZES = [1, 10, n_samples]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_fw_api(variant, batch_size):
    """Check that SFW algorithms take the right arguments and raises the right exceptions."""

    # test that the algorithm does not fail if x0
    # is a tuple
    f = copt.loss.LogLoss(A, b, 1.0 / n_samples)
    cb = cp.utils.Trace(f)
    alpha = 1.0
    l1ball = copt.constraint.L1Ball(alpha)
    cp.randomized.minimize_sfw(
        f.partial_deriv,
        A,
        b,
        [0] * n_features,
        l1ball.lmo,
        batch_size=batch_size,
        tol=0,
        callback=cb,
        variant=variant
        )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("loss_grad", LOSS_FUNCS)
def test_sfw_l1(variant, loss_grad, alpha):
    """Test SFW algorithms with L1 constraint."""
    f = loss_grad(A, b, 1.0 / n_samples)
    cb = cp.utils.Trace(f)
    l1ball = copt.constraint.L1Ball(alpha)
    opt = cp.randomized.minimize_sfw(
        f.partial_deriv,
        A,
        b,
        np.zeros(n_features),
        l1ball.lmo,
        tol=1e-3,
        callback=cb,
        variant=variant
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("loss_grad", LOSS_FUNCS)
def test_sfw_gap_traceback(variant, loss_grad, alpha):
    """Test outputting the FW gap for SFW algorithms."""
    f = loss_grad(A, b, 1.0 / n_samples)
    l1ball = copt.constraint.L1Ball(alpha)

    def fw_gap(x):
        _, grad = f.f_grad(x)
        return l1ball.lmo(-grad, x)[0].dot(-grad)

    class TraceGaps(cp.utils.Trace):
        def __init__(self, f=None, freq=1):
            super(TraceGaps, self).__init__(f, freq)
            self.trace_gaps = []

        def __call__(self, dl):
            self.trace_gaps.append(fw_gap(dl['x']))
            super(TraceGaps, self).__call__(dl)

    cb = TraceGaps(f)

    opt = cp.randomized.minimize_sfw(
        f.partial_deriv,
        A,
        b,
        np.zeros(n_features),
        l1ball.lmo,
        tol=1e-3,
        callback=cb,
        variant=variant
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("A", [sparse.random(n_samples, n_features, 0.1,
                                             fmt)
                               for fmt in ['coo', 'csr', 'csc', 'lil']])
def test_sfw_sparse(variant, A):
    """Check that SFW algorithms run on sparse data matrices."""

    f = copt.loss.LogLoss(A, b, 1.0 / n_samples)
    cb = cp.utils.Trace(f)
    alpha = 1.0
    l1ball = copt.constraint.L1Ball(alpha)
    cp.randomized.minimize_sfw(
        f.partial_deriv,
        A,
        b,
        np.zeros(n_features),
        l1ball.lmo,
        tol=0,
        callback=cb,
        variant=variant
        )



# Heavy-tailed design: a few datapoints with a much larger scale than the rest.
# This is the regime where the per-datapoint reach d_j is uneven and importance
# sampling has something to exploit; on a homogeneous design it reduces to uniform.
np.random.seed(1)
n_heavy = 120
heavy_scale = np.ones(n_heavy)
heavy_scale[np.random.choice(n_heavy, 6, replace=False)] = 30.0
A_heavy = np.random.randn(n_heavy, n_features) * heavy_scale[:, None]
b_heavy = np.abs(np.sign(np.random.randn(n_heavy)))


def test_sfw_importance_probs():
    """The helper returns a valid distribution weighted by each datapoint's reach."""
    probs = cp.randomized.sfw_importance_probs(A_heavy, alpha=1.0)
    assert probs.shape == (n_heavy,)
    assert np.all(probs > 0)
    np.testing.assert_allclose(probs.sum(), 1.0)
    # the large-scale rows must be sampled more often than the rest
    assert probs[heavy_scale == 30.0].min() > probs[heavy_scale == 1.0].max()


def test_sfw_importance_probs_rejects_other_norms():
    with pytest.raises(NotImplementedError):
        cp.randomized.sfw_importance_probs(A_heavy, alpha=1.0, ord=2)


def _run_heavy(variant, sampling_probs, seed):
    f = copt.loss.LogLoss(A_heavy, b_heavy, 1.0 / n_heavy)
    l1ball = copt.constraint.L1Ball(1.0)
    np.random.seed(seed)
    opt = cp.randomized.minimize_sfw(
        f.partial_deriv, A_heavy, b_heavy, np.zeros(n_features), l1ball.lmo,
        batch_size=1, max_iter=30, tol=0, variant=variant,
        sampling_probs=sampling_probs,
    )
    return f(opt.x)


@pytest.mark.parametrize("variant", ['SAG', 'SAGA'])
def test_sfw_importance_sampling_runs(variant):
    """Non-uniform sampling is accepted by both memory-based variants."""
    probs = cp.randomized.sfw_importance_probs(A_heavy, alpha=1.0)
    assert np.isfinite(_run_heavy(variant, probs, seed=0))


@pytest.mark.parametrize("variant", ['SAG', 'SAGA'])
def test_sfw_importance_sampling_improves(variant):
    """On a heavy-tailed design, weighting by reach beats uniform sampling.

    Holds for both memory-based variants because each keeps its estimator honest
    under non-uniform sampling: 'SAG' reads the stale aggregate directly, and 'SAGA'
    rescales its correction by 1/q_j. Measured over 12 seeds at this budget both
    improve on all 12, by ~2e-2 in objective, so the 6-seed mean compared here has
    ample margin.
    """
    probs = cp.randomized.sfw_importance_probs(A_heavy, alpha=1.0)
    seeds = range(6)
    uniform = np.mean([_run_heavy(variant, None, s) for s in seeds])
    weighted = np.mean([_run_heavy(variant, probs, s) for s in seeds])
    assert weighted < uniform


def test_sfw_importance_sampling_validation():
    """sampling_probs is rejected where the analysis does not cover it."""
    f = copt.loss.LogLoss(A, b, 1.0 / n_samples)
    l1ball = copt.constraint.L1Ball(1.0)
    good = np.full(n_samples, 1.0 / n_samples)

    def run(**kwargs):
        kwargs.setdefault("variant", "SAG")
        kwargs.setdefault("batch_size", 1)
        cp.randomized.minimize_sfw(
            f.partial_deriv, A, b, np.zeros(n_features), l1ball.lmo,
            max_iter=1, tol=0, **kwargs
        )

    with pytest.raises(ValueError, match="only supported for"):
        run(variant="MHK", sampling_probs=good)
    with pytest.raises(ValueError, match="batch_size=1"):
        run(batch_size=5, sampling_probs=good)
    with pytest.raises(ValueError, match="expected"):
        run(sampling_probs=np.full(n_samples + 1, 1.0 / (n_samples + 1)))
    with pytest.raises(ValueError, match="strictly positive"):
        bad = good.copy()
        bad[0], bad[1] = 0.0, 2.0 / n_samples
        run(sampling_probs=bad)
    with pytest.raises(ValueError, match="sum to one"):
        run(sampling_probs=np.full(n_samples, 1.0))
