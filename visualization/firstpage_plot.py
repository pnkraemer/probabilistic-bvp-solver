"""Template to turn the .csv files in data/ into work-precision plots."""

import matplotlib.pyplot as plt
import numpy as np
from _styles import LINESTYLES, MARKERS
from probnumeval.timeseries import chi2_confidence_intervals

x = np.load("./data/firstpage_plot/x.npy")
y = np.load("./data/firstpage_plot/y.npy")


plt.style.use(
    [
        "./visualization/stylesheets/fontsize/7pt.mplstyle",
        "./visualization/stylesheets/figsize/neurips/one_of_13_tile.mplstyle",
        "./visualization/stylesheets/misc/thin_lines.mplstyle",
        "./visualization/stylesheets/misc/bottomleftaxes.mplstyle",
    ]
)


fig, ax = plt.subplots(dpi=300, constrained_layout=True)
ax.spines["left"].set_position(("outward", 2))
ax.spines["bottom"].set_position(("outward", 2))


ax.plot(x, y[1], "--", color="C1", linewidth=1.5)
ax.plot(x[[0, -1]], y[0][[0, -1]], "o", color="C0")
ax.plot(x, y[0], color="C0", linewidth=1.5)
ax.annotate(
    "Solution",
    (0.5, y[0][50] + 0.3),
    color="C0",
    bbox={"facecolor": "white", "edgecolor": "white", "pad": 2},
    zorder=10,
)
ax.annotate(
    "Derivative",
    (0.5, y[1][100] + 0.3),
    color="C1",
    bbox={"facecolor": "white", "edgecolor": "white", "pad": 2},
    zorder=10,
)
# ax.set_title(r"$\bf A$" + "  ", loc="left", fontweight="bold", ha="right")

ax.set_xticks((0.0, 1.5))
ax.set_yticks((-np.pi / 2, np.pi / 2))
ax.set_xlabel("Time")
ax.set_ylabel(r"Solution / Derivative")
plt.savefig("./figures/firstpage.pdf")
plt.show()

# ax[0].plot(evalgrid, truth, linestyle="dashed", color="gray")
# ax[0].plot(evalgrid, smoother_guesses, color="C0")
# ax[0].plot(
#     initial_grid,
#     initial_guess[:, 1],
#     linestyle="None",
#     marker="o",
#     color="C0",
#     zorder=-1,
# )
# ax[1].plot(evalgrid, truth, linestyle="dashed", color="gray")
# ax[1].plot(evalgrid, smoother_ode, color="C0")


# ax[0].set_title("Via Vector")
# ax[0].set_title(r"$\bf A$" + "  ", loc="left", fontweight="bold", ha="right")

# ax[1].set_title("Via ODE")
# ax[1].set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")

# ax[0].set_ylabel(r"Derivative, $\dot y(t)$")
# ax[0].set_xlabel(r"Time, $t$")
# ax[1].set_xlabel(r"Time, $t$")


# plt.savefig("./figures/initialisation_visualisation.pdf")

# plt.show()


# fig, ax = plt.subplots(ncols=2, dpi=150, constrained_layout=True)


# for colidx, linestyle, marker in zip(results_rmse.columns[:1], LINESTYLES, MARKERS):
#     ax[0].loglog(
#         results_rmse.index,
#         results_rmse[colidx],
#         label=r"$\nu=4$",
#         linestyle=linestyle,
#         marker=marker,
#     )
#     ax[1].loglog(
#         results_anees.index,
#         results_anees[colidx],
#         linestyle=linestyle,
#         marker=marker,
#         label=r"$\nu=4$",
#     )
#     # ax[2].semilogx(results_nci.index, results_nci[colidx], marker="o", label="q=4")

#     ax[0].loglog(
#         results_rmse2.index, results_rmse2[colidx], marker=marker, label=r"$\nu=3$"
#     )
#     ax[1].loglog(
#         results_anees2.index, results_anees2[colidx], marker=marker, label=r"$\nu=3$"
#     )
#     # ax[2].semilogx(results_nci2.index, results_nci2[colidx], marker="o", label="q=3")
# ax[0].grid(which="minor")
# ax[1].grid(which="minor")


# ax[1].axhspan(out[0], out[1], alpha=0.1, color="black", linewidth=0.0)
# ax[1].axhline(1.0, color="black", linewidth=0.5)
# # ax[1].fill_between(
# #     results_anees.index, out[0], out[1], color="green", alpha=0.25, label="99% Conf."
# # )


# for axis in ax:
#     axis.set_xlabel(r"Mesh-size, $N$")
#     axis.legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)


# ax[0].set_ylabel(r"RMSE, $\varepsilon$")
# ax[1].set_ylabel(r"ANEES, $\chi^2$")

# ax[0].set_title(r"$\bf A$" + "  ", loc="left", fontweight="bold", ha="right")
# ax[1].set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")

# # ax[2].set_title("NCI")
# plt.savefig("figures/r_example_results.pdf")
# plt.show()
