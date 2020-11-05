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
# .. alpha is the regularization parameter ..
# .. which has been chosen to give 10% feature sparsity ..
import copt.constraint
import copt.loss

datasets = (
    {
        "name": "madelon",
        "loader": cp.datasets.load_madelon,
        "alpha": 1e4,
        "max_iter": 5000,
        "f_star": 0.0,
    },
    {
        "name": "gisette",
        "loader": cp.datasets.load_gisette,
        "alpha": 1e4,
        "max_iter": 5000,
        "f_star": 2.293654421822428,
    },
    {
        "name": "covtype",
        "loader": cp.datasets.load_covtype,
        "alpha": 1e4,
        "max_iter": 5000,
        "f_star": 0,
    },
    {
        "name": "RCV1",
        "loader": cp.datasets.load_rcv1,
        "alpha": 1e3,
        "max_iter": 5000,
        "f_star": 0.3114744279728717,
    },
)


variants_fw = [
    ["backtracking", "backtracking line-search"],
    ["DR", "Lipschitz step-size"],
]

for d in datasets:
    plt.figure()
    print(f"Running on the {d['name']} dataset.")

    X, y = d["loader"]()
    print(X.shape)
    n_samples, n_features = X.shape

    l1_ball = copt.constraint.L1Ball(d["alpha"])
    f = copt.loss.LogLoss(X, y)
    x0 = np.zeros(n_features)
    x0[0] = d["alpha"]  # start from a (random) vertex

    for step, label in variants_fw:

        cb = cp.utils.Trace(f)
        sol = cp.minimize_frank_wolfe(
            f.f_grad,
            x0,
            l1_ball.lmo_pairwise,
            variant='pairwise',
            x0_rep=(1., 0),
            callback=cb,
            step=step,
            lipschitz=f.lipschitz,
            max_iter=d["max_iter"],
            verbose=True,
            tol=0,
        )

        plt.plot(
            cb.trace_time,
            np.array(cb.trace_fx) - d["f_star"],
            label=label,
            markevery=10,
        )

        print("Sparsity of solution: %s" % np.mean(np.abs(sol.x) > 1e-8))
        print(f(sol.x))
    plt.legend()
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Objective function")
    plt.yscale("log")
    plt.title(d["name"])
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    #    plt.xlim((0, 0.7 * cb.trace_time[-1]))  # for aesthetics
    plt.grid()
    plt.savefig(f"figures/pairwise_benchmark_{d['name']}.png")
    plt.show()
