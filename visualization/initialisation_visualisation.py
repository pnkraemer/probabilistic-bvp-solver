"""Template to turn the .csv files in data/ into work-precision plots."""

import matplotlib.pyplot as plt
from probnumeval.timeseries import chi2_confidence_intervals
from _styles import LINESTYLES, MARKERS

import numpy as np

evalgrid = np.load("./data/initialisation_visualisation/evalgrid.npy")


initial_grid = np.load("./data/initialisation_visualisation/initial_grid.npy")


initial_guess = np.load(
    "./data/initialisation_visualisation/initial_guess.npy",
)


truth = np.load("./data/initialisation_visualisation/truth.npy")


smoother_ode = np.load("./data/initialisation_visualisation/smoother_ode.npy")


filter_ode = np.load("./data/initialisation_visualisation/filter_ode.npy")


smoother_guesses = np.load("./data/initialisation_visualisation/smoother_guesses.npy")


filter_guesses = np.load("./data/initialisation_visualisation/filter_guesses.npy")


plt.style.use(
    [
        "./visualization/stylesheets/science.mplstyle",
        "./visualization/stylesheets/misc/grid.mplstyle",
        # "./visualization/stylesheets/color/high-contrast.mplstyle",
        "./visualization/stylesheets/one_of_12_tile.mplstyle",
        "./visualization/stylesheets/9pt.mplstyle",
        "./visualization/stylesheets/hollow_markers.mplstyle",
        "./visualization/stylesheets/probnum_colors.mplstyle",
    ]
)

fig, ax = plt.subplots(ncols=1, constrained_layout=True, sharey=True)


ax.plot(evalgrid, truth, linestyle="dashed", color="gray")
ax.plot(
    evalgrid, smoother_guesses, color="C1", label="Via Vector", linewidth=2, alpha=0.5
)
ax.plot(
    initial_grid,
    initial_guess[:, 1],
    linestyle="None",
    marker=".",
    color="C1",
    zorder=-1,
)
ax.plot(evalgrid, smoother_ode, color="C0", label="Via ODE", linewidth=2, alpha=0.5)


# # ax.set_title("Via Vector")
# ax.set_title(r"$\bf A$" + "  ", loc="left", fontweight="bold", ha="right")

# # ax[1].set_title("Via ODE")
# ax.set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")

ax.set_ylabel(r"Derivative $\dot y(t)$")
ax.set_xlabel(r"Time $t$")
ax.set_ylim((-1.5, 3.5))
plt.legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)


plt.savefig("./figures/initialisation_visualisation.pdf")

plt.show()


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
