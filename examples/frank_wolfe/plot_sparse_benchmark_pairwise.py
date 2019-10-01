# python3
"""
Benchmark of Pairwise Frank-Wolfe variants for sparse logistic regression
=========================================================================

Speed of convergence of different Frank-Wolfe variants on various
problems with a logistic regression loss (:meth:`copt.utils.LogLoss`)
and a L1 ball constraint (:meth:`copt.utils.L1Ball`).
"""
import matplotlib.pyplot as plt
import numpy as np
import copt as cp

# .. datasets and their loading functions ..
# .. last value si the regularization parameter ..
# .. which has been chosen to give 10% feature sparsity ..
datasets = (
    {
        "name": "RCV1",
        "loader": cp.datasets.load_rcv1,
        "alpha": 1e3,
        "max_iter": 500,
        "f_star": 0.3114744279728717,
    },
    {
        "name": "gisette",
        "loader": cp.datasets.load_gisette,
        "alpha": 1e4,
        "max_iter": 500,
        "f_star": 2.293654421822428,
    },
    {
        "name": "madelon",
        "loader": cp.datasets.load_madelon,
        "alpha": 1e4,
        "max_iter": 500,
        "f_star": 0.0,
    },
    {
        "name": "covtype",
        "loader": cp.datasets.load_covtype,
        "alpha": 1e4,
        "max_iter": 500,
        "f_star": 0,
    },
)


variants_fw = [
    ["adaptive", "adaptive step-size", "s"],
    # ["adaptive2+", "linesearch+ step-size", "s"],
    # ["adaptive3", "adaptive3 step-size", "+"],
    # ["adaptive4", "adaptive4 step-size", "x"],
    ["DR", "Lipschitz step-size", "<"],
    ["adaptive_scipy", "scipy linesearch step-size", "^"],
    ["panj", "panj step-size", ">"],
]

for d in datasets:
    plt.figure()
    print("Running on the %s dataset" % d["name"])

    X, y = d["loader"]()
    print(X.shape)
    n_samples, n_features = X.shape

    l1_ball = cp.utils.L1Ball(d["alpha"])
    f = cp.utils.LogLoss(X, y)
    x0 = np.zeros(n_features)
    x0[0] = d["alpha"]  # start from a (random) vertex
    active_set = np.zeros(n_features * 2)
    active_set[0] = 1

    for step_size, label, marker in variants_fw:

        cb = cp.utils.Trace(f)
        sol = cp.minimize_pairwise_frank_wolfe(
            f.f_grad,
            x0,
            active_set,
            l1_ball.lmo_pairwise,
            callback=cb,
            step_size=step_size,
            lipschitz=f.lipschitz,
            max_iter=d["max_iter"],
            verbose=True,
            tol=0,
        )

        plt.plot(
            cb.trace_time,
            np.array(cb.trace_fx) - d["f_star"],
            label=label,
            marker=marker,
            markevery=10,
        )

    print("Sparsity of solution: %s" % np.mean(np.abs(sol.x) > 1e-8))
    print(f(sol.x))
    plt.legend()
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Objective function")
    plt.title(d["name"])
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    #    plt.xlim((0, 0.7 * cb.trace_time[-1]))  # for aesthetics
    plt.grid()
    plt.show()
