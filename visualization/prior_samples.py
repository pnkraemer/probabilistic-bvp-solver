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
        "./visualization/stylesheets/color/vibrant.mplstyle",
    ]
)

orders = [3]
num_samples = 10
colors = ["black", "blue"]
styles = ["--", "-"]

PATH = "./data/prior_samples/samples_"
grid = np.load(PATH + "grid.npy")
fig, axes = plt.subplots(ncols=1, dpi=200, constrained_layout=True)
for q, col, linestyle in zip(orders, colors, styles):
    # for q, ax, col in zip(orders, axes, colors):

    for idx in range(num_samples):
        samples = np.load(PATH + str(q) + str(idx) + ".npy")

        if idx == 0:
            axes.plot(
                grid,
                samples[:, 0],
                color=col,
                alpha=0.75,
                linestyle=linestyle,
                label=f"$\\nu = {q}$",
            )
        else:
            axes.plot(
                grid,
                samples[:, 0],
                color=col,
                alpha=0.75,
                linestyle=linestyle,
                label=f"_$\\nu={q}$",
            )
axes.set_xlabel(r"Time $t$")
# axes.set_title(f"Order $\\nu = {q}$")
axes.set_xticks([])
axes.set_yticks([])
axes.set_title(r"$\bf D$", loc="left", fontweight="bold")
# axes.set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")
axes.set_ylabel(r"Prior $Y_0(t)$")

# plt.legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)


plt.savefig("./figures/prior_samples2.pdf")

plt.show()
