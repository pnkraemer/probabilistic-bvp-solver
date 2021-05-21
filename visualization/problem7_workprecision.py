"""Plot WP on bratu's problem."""
import json

import matplotlib.pyplot as plt
import numpy as np
from probnumeval.timeseries import chi2_confidence_intervals
from math import log10, floor
def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))
out = chi2_confidence_intervals(dim=1, perc=0.95)

with open("./data/problem7_problem_work_precision.json", "r") as infile:
    data = json.load(infile)
# plt.style.use(
#     [
#   "./visualization/stylesheets/science.mplstyle",
#   "./visualization/stylesheets/misc/grid.mplstyle",
#   "./visualization/stylesheets/color/muted.mplstyle",
#   "./visualization/stylesheets/13_tile_jmlr.mplstyle",
#   "./visualization/stylesheets/9pt.mplstyle",
# ]
# )
print(data)


plt.style.use(
    [
        "./visualization/stylesheets/fontsize/7pt.mplstyle",
        "./visualization/stylesheets/figsize/neurips/13_tile.mplstyle",
        "./visualization/stylesheets/misc/thin_lines.mplstyle",
        "./visualization/stylesheets/misc/bottomleftaxes.mplstyle",
        "./visualization/stylesheets/marker/framed_markers.mplstyle",
        # "./visualization/stylesheets/color/probnum_colors.mplstyle",
    ]
)


fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, dpi=300)

xticks = []
xticklabels = []
for problem_index, key in enumerate(data.keys()):
    problem_results = data[key]["4"]
    problem_results_1e2 = problem_results["0.01"]
    problem_results_1e5 = problem_results["1e-05"]

    error_1e2 = problem_results_1e2["error"]
    N_1e2 = problem_results_1e2["N"]
    time_1e2 = problem_results_1e2["time"]
    scipy_error_1e2 = problem_results_1e2["error_scipy"]
    scipy_N_1e2 = problem_results_1e2["N_scipy"]
    scipy_time_1e2 = problem_results_1e2["time_scipy"]

    error_1e5 = problem_results_1e5["error"]
    N_1e5 = problem_results_1e5["N"]
    time_1e5 = problem_results_1e5["time"]
    scipy_error_1e5 = problem_results_1e5["error_scipy"]
    scipy_N_1e5 = problem_results_1e5["N_scipy"]
    scipy_time_1e5 = problem_results_1e5["time_scipy"]

    xticks.append(problem_index)
    xticklabels.append(key)
    SHIFT = 0.05
    ax.annotate(
        f"({N_1e2}, {round_to_1(time_1e2)})",
        (problem_index - 0.4, error_1e2),
        zorder=10,
        bbox={"facecolor": "white", "edgecolor": "white", "pad": 0},
        fontsize="small"
    )
    ax.annotate(
        f"({N_1e5}, {round_to_1(time_1e5)})",
        (problem_index - 0.4, error_1e5),
        zorder=10,
        bbox={"facecolor": "white", "edgecolor": "white", "pad": 0},
        fontsize="small"
    )
    ax.annotate(
        f"({scipy_N_1e2}, {round_to_1(scipy_time_1e2)})",
        (problem_index + 0.1, scipy_error_1e2),
        zorder=10,
        bbox={"facecolor": "white", "edgecolor": "white", "pad": 0},
        fontsize="small"    
    )
    ax.annotate(
        f"({scipy_N_1e5}, {round_to_1(scipy_time_1e5)})",
        (problem_index + 0.1, scipy_error_1e5),
        zorder=10,
        bbox={"facecolor": "white", "edgecolor": "white", "pad": 0},
        fontsize="small"
    )

    ax.semilogy(problem_index - SHIFT, error_1e2, marker="P", color="C0")
    ax.semilogy(problem_index - SHIFT, error_1e5, marker="P", color="C1")
    ax.semilogy(problem_index + SHIFT, scipy_error_1e2, marker="d", color="C0")
    ax.semilogy(problem_index + SHIFT, scipy_error_1e5, marker="d", color="C1")
    #
    # ax.vlines(
    #     x=problem_index - SHIFT,
    #     ymin=np.minimum(error_1e2, 1e-2),
    #     ymax=np.maximum(error_1e2, 1e-2),
    #     linestyle="-",
    #     linewidth=2.5,
    #     alpha=0.75,
    #     color="C0",
    # )
    # ax.vlines(
    #     x=problem_index - SHIFT,
    #     ymin=np.minimum(error_1e5, 1e-5),
    #     ymax=np.maximum(error_1e5, 1e-5),
    #     linestyle="-",
    #     linewidth=2.5,
    #     alpha=0.75,
    #     color="C1",
    # )
    #
    # ax.vlines(
    #     x=problem_index + SHIFT,
    #     ymin=np.minimum(scipy_error_1e2, 1e-2),
    #     ymax=np.maximum(scipy_error_1e2, 1e-2),
    #     linestyle="dashed",
    #     linewidth=4.5,
    #     alpha=0.5,
    #     color="C0",
    # )
    # ax.vlines(
    #     x=problem_index + SHIFT,
    #     ymin=np.minimum(scipy_error_1e5, 1e-5),
    #     ymax=np.maximum(scipy_error_1e5, 1e-5),
    #     linestyle="dashed",
    #     linewidth=4.5,
    #     alpha=0.5,
    #     color="C1",
    # )

# print(problem_index, key)
plt.legend()
ax.annotate("Tol $10^{-2}$", (-0.1, 2e-2), color="C0")
ax.annotate("Tol $10^{-5}$", (0.5, 2e-5), color="C1")
ax.axhline(1e-2, linestyle="dashed", color="C0", linewidth=1)
ax.axhline(1e-5, linestyle="dashed", color="C1", linewidth=1)
ax.set_xlim(np.amin(xticks) - 0.5, np.amax(xticks) + 0.5)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_ylabel("RMSE")

plt.savefig("./figures/all_errors.pdf")

plt.show()

#
#
# fig, ax = plt.subplots(ncols=3, nrows=2, dpi=150, constrained_layout=True)
#
# for axis in ax.flatten():
#     axis.spines["left"].set_position(("outward", 2))
#     axis.spines["bottom"].set_position(("outward", 2))
#
# LINEWIDTH = 1.0
# ALPHA = 0.9
#
# markers = ["s", "d", "^", "o"]
# colors = ["C0", "C1", "C2", "C3"]
# for q, marker, color in zip(data.keys(), markers, colors):
#     qres = data[q]
#     Ns = []
#     chi2s = []
#     errors = []
#     tols = []
#     times = []
#     hs = []
#
#     scipy_errors = []
#     scipy_times = []
#     scipy_Ns = []
#     scipy_hs = []
#     for tol in qres.keys():
#         qtolres = qres[tol]
#
#         chi2 = qtolres["chi2"]
#         error = qtolres["error"]
#         N = qtolres["N"]
#         time = qtolres["time"]
#         h = qtolres["largest_step"]
#
#         hs.append(h)
#         times.append(time)
#         tols.append(float(tol))
#         Ns.append(N)
#         chi2s.append(chi2)
#         errors.append(error)
#
#         scipy_error = qtolres["error_scipy"]
#         scipy_time = qtolres["time_scipy"]
#         scipy_N = qtolres["N_scipy"]
#         scipy_h = qtolres["largest_step_scipy"]
#
#         scipy_errors.append(scipy_error)
#         scipy_times.append(scipy_time)
#         scipy_Ns.append(scipy_N)
#         scipy_hs.append(scipy_h)
#
#     ax[0][0].loglog(
#         Ns,
#         errors,
#         color=color,
#         label=f"$\\nu={q}$",
#         marker=marker,
#         markersize=4,
#         linewidth=LINEWIDTH,
#         alpha=ALPHA,
#     )
#     ax[0][1].loglog(
#         hs,
#         errors,
#         color=color,
#         label=f"$\\nu={q}$",
#         marker=marker,
#         markersize=4,
#         linewidth=LINEWIDTH,
#         alpha=ALPHA,
#     )
#     ax[0][2].loglog(
#         times,
#         errors,
#         color=color,
#         label=f"$\\nu={q}$",
#         marker=marker,
#         markersize=4,
#         linewidth=LINEWIDTH,
#         alpha=ALPHA,
#     )
#
#     ax[1][0].loglog(
#         tols,
#         np.array(tols),
#         color="gray",
#         linestyle="dotted",
#     )
#     ax[1][0].loglog(
#         tols,
#         errors,
#         color=color,
#         label=f"$\\nu={q}$",
#         marker=marker,
#         markersize=4,
#         linewidth=LINEWIDTH,
#         alpha=ALPHA,
#     )
#
#     ax[1][1].loglog(
#         times,
#         Ns,
#         color=color,
#         label=f"$\\nu={q}$",
#         marker=marker,
#         markersize=4,
#         linewidth=LINEWIDTH,
#         alpha=ALPHA,
#     )
#
#     ax[1][2].loglog(
#         Ns,
#         chi2s,
#         label=f"$\\nu={q}$",
#         marker=marker,
#         markersize=4,
#         color=color,
#         linewidth=LINEWIDTH,
#         alpha=ALPHA,
#     )
#
#
# ax[1][0].loglog(
#     tols,
#     scipy_errors,
#     color="darkgray",
#     marker="o",
#     markersize=4,
#     linewidth=1.0,
#     alpha=0.8,
#     label="SciPy",
# )
# ax[0][0].loglog(
#     scipy_Ns,
#     scipy_errors,
#     color="darkgray",
#     marker="o",
#     markersize=4,
#     linewidth=1.0,
#     alpha=0.8,
#     label="SciPy",
# )
# ax[0][1].loglog(
#     scipy_hs,
#     scipy_errors,
#     color="darkgray",
#     marker="o",
#     markersize=4,
#     linewidth=1.0,
#     alpha=0.8,
#     label="SciPy",
# )
# ax[0][2].loglog(
#     scipy_times,
#     scipy_errors,
#     color="darkgray",
#     marker="o",
#     markersize=4,
#     linewidth=1.0,
#     alpha=0.8,
#     label="SciPy",
# )
# ax[1][1].loglog(
#     scipy_times,
#     scipy_Ns,
#     color="darkgray",
#     marker="o",
#     markersize=4,
#     linewidth=1.0,
#     alpha=0.9,
#     label="SciPy",
# )
#
# ax[0][0].set_xlabel("No. of grid points")
# ax[0][0].set_ylabel("RMSE")
#
# ax[0][1].set_xlabel("Largest step")
# ax[0][1].set_ylabel("RMSE")
#
# ax[0][2].set_xlabel("Runtime (s)")
# ax[0][2].set_ylabel("RMSE")
#
# ax[1][0].set_xlabel("Tolerance")
# ax[1][0].set_ylabel("RMSE")
#
# ax[1][1].set_xlabel("Runtime (s)")
# ax[1][1].set_ylabel("No. of grid points")
#
# ax[1][2].set_xlabel("No. of grid points")
# ax[1][2].set_ylabel("$\\chi^2$ statistic")
#
# # ax[0].set_xlabel(r"Tolerance")
# # ax[1].set_xlabel(r"Runtime (s)")
# # ax[2].set_xlabel("Runtime (s)")
# # ax[3].set_xlabel(r"No. of grid points")
# #
# # ax[0].set_ylabel("RMSE")
# # ax[1].set_ylabel("RMSE")
# # ax[2].set_ylabel(r"No. of grid points")
# # ax[3].set_ylabel("ANEES")
#
# ax[1][2].axhspan(out[0], out[1], alpha=0.1, color="black", linewidth=0.0)
# ax[1][2].axhline(1.0, color="black", linewidth=0.5)
#
# ax[0][2].legend(
#     fancybox=False, edgecolor="black", fontsize="x-small"
# ).get_frame().set_linewidth(0.5)
# # ax[1].legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)
# # ax[2].legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)
#
# plt.savefig("./figures/problem7_workprecision.pdf")
