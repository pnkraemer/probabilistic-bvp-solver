"""Template to turn the .csv files in data/ into work-precision plots."""

import matplotlib.pyplot as plt
from probnumeval.timeseries import chi2_confidence_intervals
from _styles import LINESTYLES, MARKERS

import numpy as np


plt.style.use(
    [
        "./visualization/stylesheets/science.mplstyle",
        "./visualization/stylesheets/misc/grid.mplstyle",
        "./visualization/stylesheets/13_tile_jmlr.mplstyle",
        "./visualization/stylesheets/10pt.mplstyle",
        "./visualization/stylesheets/probnum_colors.mplstyle",
    ]
)

orders = [1, 5]
num_samples = 15
colors = ["C0", "C1"]

PATH = "./data/prior_samples/samples_"
grid = np.load(PATH + "grid.npy")
fig, axes = plt.subplots(ncols=2, dpi=350, constrained_layout=True, sharey=True)
for q, ax, col in zip(orders, axes, colors):

    for idx in range(num_samples):
        samples = np.load(PATH + str(q) + str(idx) + ".npy")

        ax.plot(grid, samples[:, 0], color=col)
        ax.set_xlabel(r"Time, $t$")
        ax.set_title(f"Order, $\\nu = {q}$")

axes[0].set_title(r"$\bf A$" + "  ", loc="left", fontweight="bold", ha="right")
axes[1].set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")
axes[0].set_ylabel(r"Prior, $Y_0(t)$")


plt.savefig("./figures/prior_samples.pdf")

plt.show()
