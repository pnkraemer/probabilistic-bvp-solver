"""Draw samples from a bridge prior."""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars
from bvps import bratus, BoundaryValueProblem, WrappedIntegrator
from tqdm import tqdm


bvp = bratus(tmax=1.0)
grid = np.linspace(bvp.t0, bvp.tmax, 20)

num_orders = 7
num_samples = 5
fig, axes = plt.subplots(
    ncols=num_orders,
    figsize=(num_orders * 3, 3),
    sharey=True,
    dpi=250,
    constrained_layout=True,
)


for q, ax in enumerate(tqdm(axes)):
    ibm = statespace.IBM(
        ordint=q,
        spatialdim=2,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )

    integ = WrappedIntegrator(ibm, bvp)

    rv = randvars.Normal(np.zeros(ibm.dimension), 2.0 * np.eye(ibm.dimension))
    base_measure_samples = np.random.randn(len(grid), ibm.dimension)

    def generate_samples(grid, transition, rv, base_measure_samples):
        rv, _ = transition.forward_rv(rv, grid[0], dt=0.0)
        smp = rv.mean + rv.cov_cholesky @ base_measure_samples[0]
        yield smp
        for t, tnew, b in zip(grid[:-1], grid[1:], base_measure_samples[1:]):
            dt = tnew - t
            rv, _ = transition.forward_realization(smp, t=t, dt=dt)
            smp = rv.mean + rv.cov_cholesky @ b
            yield smp

    base_measure_samples = np.random.randn(num_samples, len(grid), ibm.dimension)

    for smp in base_measure_samples:
        samples = np.array(list(generate_samples(grid, integ, rv, smp)))
        ax.plot(grid, samples[:, 0], color="black", linestyle="dashed")
        ax.set_title(f"Order: $\\nu={q}$")
        ax.set_xlabel("Time, $t$")
axes[0].set_ylabel("Prior, $Y_0(t)$")
plt.savefig("figures/IBMBridges.pdf")
plt.show()

# for t, smp in zip(grid, samples):
#     pass
