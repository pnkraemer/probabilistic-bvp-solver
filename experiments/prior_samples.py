"""Draw samples from a bridge prior."""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars
from bvps import bratus, BoundaryValueProblem, WrappedIntegrator
from tqdm import tqdm


bvp = bratus(tmax=1.0)
grid = np.linspace(bvp.t0, bvp.tmax, 20)

orders = [2, 3, 5, 8]
num_samples = 10
fig, axes = plt.subplots(
    ncols=len(orders),
    nrows=2,
    figsize=(len(orders) * 2, 4),
    sharey="row",
    sharex="col",
    dpi=250,
    constrained_layout=True,
)


def generate_samples(grid, transition, rv, base_measure_samples, fix=True):
    if fix:
        rv, _ = transition.forward_rv(rv, grid[0], dt=0.0)
    smp = rv.mean + rv.cov_cholesky @ base_measure_samples[0]
    yield smp
    for t, tnew, b in zip(grid[:-1], grid[1:], base_measure_samples[1:]):
        dt = tnew - t
        rv, _ = transition.forward_realization(smp, t=t, dt=dt)
        smp = rv.mean + rv.cov_cholesky @ b
        yield smp


for q, ax in tqdm(zip(orders, axes.T), total=len(orders)):
    ibm = statespace.IBM(
        ordint=q,
        spatialdim=2,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )

    integ = WrappedIntegrator(ibm, bvp)

    rv = randvars.Normal(np.zeros(ibm.dimension), 2.0 * np.eye(ibm.dimension))
    rv2 = randvars.Normal(np.zeros(ibm.dimension), 2.0 * np.eye(ibm.dimension))

    base_measure_samples = np.random.randn(num_samples, len(grid), ibm.dimension)
    base_measure_samples2 = np.random.randn(num_samples, len(grid), ibm.dimension)

    for smp, smp2 in zip(base_measure_samples, base_measure_samples2):
        samples = np.array(list(generate_samples(grid, integ, rv, smp)))
        samples2 = np.array(list(generate_samples(grid, integ, rv2, smp2, fix=False)))
        ax[0].plot(
            grid, samples[:, 0], color="darkorange", linestyle="dashed", linewidth=0.75
        )
        ax[0].set_title(f"Order: $\\nu={q}$")

        ax[1].plot(
            grid, samples2[:, 0], color="teal", linestyle="dashed", linewidth=0.75
        )
        ax[1].set_xlabel("Time, $t$")
axes[0][0].set_ylabel("Type I Samples")
axes[1][0].set_ylabel("Type II Samples")
plt.savefig("figures/IBMBridges.pdf")
plt.show()

# for t, smp in zip(grid, samples):
#     pass
