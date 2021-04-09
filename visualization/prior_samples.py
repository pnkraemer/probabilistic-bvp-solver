"""Template to turn the .csv files in data/ into work-precision plots."""

import matplotlib.pyplot as plt
from probnumeval.timeseries import chi2_confidence_intervals
from _styles import LINESTYLES, MARKERS

import numpy as np


plt.style.use(
    [
        "./probabilistic-bvp-solver/visualization/science.mplstyle",
        "./probabilistic-bvp-solver/visualization/misc/grid.mplstyle",
        "./probabilistic-bvp-solver/visualization/color/high-contrast.mplstyle",
        "./probabilistic-bvp-solver/stylesheets/13_tile_jmlr.mplstyle",
        "./probabilistic-bvp-solver/stylesheets/10pt.mplstyle",
        "./probabilistic-bvp-solver/stylesheets/hollow_markers.mplstyle",
        # "./stylesheets/probnum_colors.mplstyle"
    ]
)
orders = [1, 3, 5]
num_samples = 15
path = "./probabilistic-bvp-solver/data/prior_samples/samples_"
grid = np.load(path + "grid.npy")
fig, axes = plt.subplots(ncols=3, dpi=350, constrained_layout=True, sharey=True)
for q, ax in zip(orders, axes):

    for idx in range(num_samples):
        samples = np.load(path + str(q) + str(idx) + ".npy")

        ax.plot(grid, samples[:, 0], color="black")
        ax.set_xlabel(r"Time, $t$")
        ax.set_title(f"Order, $\\nu = {q}$")

axes[0].set_title(r"$\bf A$" + "  ", loc="left", fontweight="bold", ha="right")
axes[1].set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")
axes[2].set_title(r"$\bf C$" + "  ", loc="left", fontweight="bold", ha="right")
axes[0].set_ylabel(r"Prior, $Y_0(t)$")

# ax[0].plot(evalgrid, truth, linestyle="dashed", color="gray")
# ax[0].plot(evalgrid, smoother_guesses)
# ax[0].plot(
#     initial_grid,
#     initial_guess[:, 1],
#     linestyle="None",
#     marker="o",
#     color="C0",
#     zorder=-1,
# )
# ax[1].plot(evalgrid, truth, linestyle="dashed", color="gray")
# ax[1].plot(evalgrid, smoother_ode)


# ax[0].set_title("Via Vector")
# ax[0].set_title(r"$\bf A$" + "  ", loc="left", fontweight="bold", ha="right")

# ax[1].set_title("Via ODE")
# ax[1].set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")

# ax[0].set_ylabel(r"Derivative, $\dot y(t)$")
# ax[0].set_xlabel(r"Time, $t$")
# ax[1].set_xlabel(r"Time, $t$")


plt.savefig("./probabilistic-bvp-solver/figures/prior_samples.pdf")

plt.show()
