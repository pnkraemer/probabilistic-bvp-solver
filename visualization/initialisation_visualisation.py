"""Template to turn the .csv files in data/ into work-precision plots."""

import matplotlib.pyplot as plt
from probnumeval.timeseries import chi2_confidence_intervals
from _styles import LINESTYLES, MARKERS

import numpy as np

evalgrid = np.load(
    "./probabilistic-bvp-solver/data/initialisation_visualisation/evalgrid.npy"
)


initial_grid = np.load(
    "./probabilistic-bvp-solver/data/initialisation_visualisation/initial_grid.npy"
)


initial_guess = np.load(
    "./probabilistic-bvp-solver/data/initialisation_visualisation/initial_guess.npy",
)


truth = np.load(
    "./probabilistic-bvp-solver/data/initialisation_visualisation/truth.npy"
)


smoother_ode = np.load(
    "./probabilistic-bvp-solver/data/initialisation_visualisation/smoother_ode.npy"
)


filter_ode = np.load(
    "./probabilistic-bvp-solver/data/initialisation_visualisation/filter_ode.npy"
)


smoother_guesses = np.load(
    "./probabilistic-bvp-solver/data/initialisation_visualisation/smoother_guesses.npy"
)


filter_guesses = np.load(
    "./probabilistic-bvp-solver/data/initialisation_visualisation/filter_guesses.npy"
)


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

fig, ax = plt.subplots(ncols=2, dpi=350, constrained_layout=True, sharey=True)


ax[0].plot(evalgrid, truth, linestyle="dashed", color="gray")
ax[0].plot(evalgrid, smoother_guesses)
ax[0].plot(
    initial_grid,
    initial_guess[:, 1],
    linestyle="None",
    marker="o",
    color="C0",
    zorder=-1,
)
ax[1].plot(evalgrid, truth, linestyle="dashed", color="gray")
ax[1].plot(evalgrid, smoother_ode)


ax[0].set_title("Via Vector")
ax[0].set_title(r"$\bf A$" + "  ", loc="left", fontweight="bold", ha="right")

ax[1].set_title("Via ODE")
ax[1].set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")

ax[0].set_ylabel(r"Derivative, $\dot y(t)$")
ax[0].set_xlabel(r"Time, $t$")
ax[1].set_xlabel(r"Time, $t$")


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
