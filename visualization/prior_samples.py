"""Template to turn the .csv files in data/ into work-precision plots."""

import matplotlib.pyplot as plt
from probnumeval.timeseries import chi2_confidence_intervals
from _styles import LINESTYLES, MARKERS

import numpy as np


plt.style.use(
    [
        "./visualization/stylesheets/science.mplstyle",
        # "./visualization/stylesheets/misc/grid.mplstyle",
        "./visualization/stylesheets/one_of_12_tile.mplstyle",
        "./visualization/stylesheets/8pt.mplstyle",
        "./visualization/stylesheets/probnum_colors.mplstyle",
    ]
)

orders = [3]
q = 3
num_samples = 10
colors = ["C0", "C1"]
styles = ["-", "-"]

PATH = "./data/prior_samples/samples_"
grid = np.load(PATH + "grid.npy")
fig, axes = plt.subplots(ncols=2, dpi=200, constrained_layout=True)

for idx in range(num_samples):
    samples_bridge = np.load(PATH + str(q) + str(idx) + ".npy")
    samples = np.load(PATH + str(q) + str(idx) + "2.npy")

    axes[1].plot(
        grid,
        samples_bridge[:, 0],
        color="C0",
        alpha=0.75,
    )
    axes[0].plot(
        grid,
        samples[:, 0],
        color="C1",
        alpha=0.75,
    )

for ax in axes:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel(r"Time $t$")
    # axes.set_title(f"Order $\\nu = {q}$")
    ax.set_xticks([])
    ax.set_yticks([])
# axes[0].set_title(r"$\bf B$", loc="left", fontweight="bold", pad=5)
# axes[1].set_title(r"$\bf C$", loc="left", fontweight="bold", pad=5)
# axes.set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")
axes[0].set_ylabel(r"Samples $Y_0(t)$")

# plt.legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)


plt.savefig("./figures/prior_samples.pdf")

plt.show()
