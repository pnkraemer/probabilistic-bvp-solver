"""Draw samples from a bridge prior."""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars
from bvps import (
    bratus,
    BoundaryValueProblem,
    WrappedIntegrator,
    generate_samples,
    r_example,
)
from tqdm import tqdm


bvp = r_example()
grid = np.linspace(bvp.t0, bvp.tmax, 200)

orders = [6, 11]
num_samples = 5
fig, axes = plt.subplots(
    ncols=len(orders),
    nrows=2,
    figsize=(len(orders) * 2, 4),
    sharey="row",
    sharex="col",
    dpi=250,
    constrained_layout=True,
)


path = "./probabilistic-bvp-solver/data/prior_samples/samples_"
np.save(path + "grid", grid)
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

    for idx, (smp, smp2) in enumerate(zip(base_measure_samples, base_measure_samples2)):
        samples = np.array(list(generate_samples(grid, integ, rv, smp)))
        np.save(path + str(q) + str(idx), samples)
        samples2 = np.array(list(generate_samples(grid, integ, rv2, smp2, fix=False)))
        ax[0].plot(
            grid, samples[:, 0], color="darkorange", linestyle="dashed", linewidth=0.75
        )
        ax[0].set_title(f"Order: $\\nu={q}$")

        # ax[1].plot(
        #     grid, samples2[:, 0], color="teal", linestyle="dashed", linewidth=0.75
        # )
        # ax[1].set_xlabel("Time, $t$")
axes[0][0].set_ylabel("Type I Samples")
axes[1][0].set_ylabel("Type II Samples")
plt.savefig("./figures/IBMBridges.pdf")
plt.show()

# for t, smp in zip(grid, samples):
#     pass
