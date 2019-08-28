# python3
"""
Benchmark of Frank-Wolfe variants for sparse logistic regression
================================================================

Speed of convergence of different Frank-Wolfe variants on various
problems with a logistic regression loss (:meth:`copt.utils.LogLoss`)
and a L1 ball constraint (:meth:`copt.utils.L1Ball`).
"""
import copt as cp
import matplotlib.pyplot as plt
import numpy as np

# .. datasets and their loading functions ..
datasets = [
    ("Gisette", cp.datasets.load_gisette),
    ("RCV1", cp.datasets.load_rcv1),
    ("Madelon", cp.datasets.load_madelon),
    ("Covtype", cp.datasets.load_covtype),
]


variants_fw = [
    ["adaptive", "adaptive step-size", "s"],
    ["adaptive_scipy", "scipy linesearch step-size", "^"],
    ["adaptive_scipy+", "linesearch+ step-size", "s"],
    # ["adaptive3", "adaptive3 step-size", "+"],
    # ["adaptive4", "adaptive4 step-size", "x"],
    ["adaptive5", "adaptive5 step-size", ">"],
    ["DR", "Lipschitz step-size", "<"],
]

for dataset_title, load_data in datasets:
    plt.figure()
    print("Running on the %s dataset" % dataset_title)

    X, y = load_data()
    n_samples, n_features = X.shape

    # the size of the constraint set. We set it to
    # (for example) n_features / 2
    l1_ball = cp.utils.L1Ball(n_features / 2.0)
    f = cp.utils.LogLoss(X, y)
    x0 = np.zeros(n_features)

    for step_size, label, marker in variants_fw:

        cb = cp.utils.Trace(f)
        cp.minimize_frank_wolfe(
            f.f_grad,
            x0,
            l1_ball.lmo,
            callback=cb,
            step_size=step_size,
            lipschitz=f.lipschitz,
            # max_iter=1000
        )

        plt.plot(cb.trace_time, cb.trace_fx, label=label, marker=marker, markevery=10)
    plt.legend()
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Objective function")
    plt.title(dataset_title)
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.xlim((0, 0.7 * cb.trace_time[-1]))  # for aesthetics
    plt.grid()
    plt.show()
