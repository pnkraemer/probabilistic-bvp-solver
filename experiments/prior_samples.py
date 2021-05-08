"""Draw samples from a bridge prior."""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars
from tqdm import tqdm

from bvps import problem_examples, bridges, generate_samples

SAVE_DATA = True
PATH = "./data/prior_samples/samples_"


bvp = problem_examples.r_example()
grid = np.linspace(bvp.t0, bvp.tmax, 100)

orders = [1, 3]
num_samples = 10
fig, axes = plt.subplots(
    ncols=len(orders),
    nrows=1,
    figsize=(len(orders) * 2, 2),
    dpi=250,
    constrained_layout=True,
)


if SAVE_DATA:
    np.save(PATH + "grid", grid)

for q, ax in tqdm(zip(orders, axes.T), total=len(orders)):
    ibm = statespace.IBM(
        ordint=q,
        spatialdim=2,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )

    prior = bridges.GaussMarkovBridge(ibm, bvp)

    initrv_not_initialised = randvars.Normal(
        np.zeros(ibm.dimension), np.eye(ibm.dimension)
    )
    initrv = prior.initialise_boundary_conditions(initrv_not_initialised)

    base_measure_samples = np.random.randn(num_samples, len(grid), ibm.dimension)
    base_measure_samples2 = np.random.randn(num_samples, len(grid), ibm.dimension)

    for idx, (smp, smp2) in enumerate(zip(base_measure_samples, base_measure_samples2)):
        sample_generator = generate_samples.generate_samples(grid, prior, initrv, smp)
        samples = np.array(list(sample_generator))

        sample_generator2 = generate_samples.generate_samples(
            grid, ibm, initrv_not_initialised, smp
        )
        samples2 = np.array(list(sample_generator2))

        if SAVE_DATA:
            np.save(PATH + str(q) + str(idx), samples)
            np.save(PATH + str(q) + str(idx) + "2", samples2)

        ax.plot(
            grid, samples[:, 0], color="darkorange", linestyle="dashed", linewidth=0.75
        )
        ax.plot(grid, samples2[:, 0], color="blue", linestyle="dashed", linewidth=0.75)
        ax.set_title(f"Order: $\\nu={q}$")

axes[0].set_ylabel("Samples")
# plt.savefig("./figures/IBMBridges.pdf")
plt.show()
