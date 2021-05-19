"""Plot WP on bratu's problem."""
import json

import matplotlib.pyplot as plt
import numpy as np
from probnumeval.timeseries import chi2_confidence_intervals

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

plt.style.use(
    [
        "./visualization/stylesheets/fontsize/7pt.mplstyle",
        "./visualization/stylesheets/figsize/neurips/23_tile.mplstyle",
        "./visualization/stylesheets/misc/thin_lines.mplstyle",
        "./visualization/stylesheets/misc/bottomleftaxes.mplstyle",
        "./visualization/stylesheets/color/probnum_colors.mplstyle",
    ]
)


fig, ax = plt.subplots(ncols=3, nrows=2, dpi=150, constrained_layout=True)

for axis in ax.flatten():
    axis.spines["left"].set_position(("outward", 2))
    axis.spines["bottom"].set_position(("outward", 2))

LINEWIDTH = 1.0
ALPHA = 0.9

markers = ["s", "d", "^", "o"]
colors = ["C0", "C1", "C2", "C3"]
for q, marker, color in zip(data.keys(), markers, colors):
    qres = data[q]
    Ns = []
    chi2s = []
    errors = []
    tols = []
    times = []
    hs = []

    scipy_errors = []
    scipy_times = []
    scipy_Ns = []
    scipy_hs = []
    for tol in qres.keys():
        qtolres = qres[tol]

        chi2 = qtolres["chi2"]
        error = qtolres["error"]
        N = qtolres["N"]
        time = qtolres["time"]
        h = qtolres["largest_step"]

        hs.append(h)
        times.append(time)
        tols.append(float(tol))
        Ns.append(N)
        chi2s.append(chi2)
        errors.append(error)

        scipy_error = qtolres["error_scipy"]
        scipy_time = qtolres["time_scipy"]
        scipy_N = qtolres["N_scipy"]
        scipy_h = qtolres["largest_step_scipy"]

        scipy_errors.append(scipy_error)
        scipy_times.append(scipy_time)
        scipy_Ns.append(scipy_N)
        scipy_hs.append(scipy_h)

    ax[0][0].loglog(
        Ns,
        errors,
        color=color,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=4,
        linewidth=LINEWIDTH,
        alpha=ALPHA,
    )
    ax[0][1].loglog(
        hs,
        errors,
        color=color,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=4,
        linewidth=LINEWIDTH,
        alpha=ALPHA,
    )
    ax[0][2].loglog(
        times,
        errors,
        color=color,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=4,
        linewidth=LINEWIDTH,
        alpha=ALPHA,
    )


    ax[1][0].loglog(
        tols,
        np.array(tols),
        color="gray",
        linestyle="dotted",
    )
    ax[1][0].loglog(
        tols,
        errors,
        color=color,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=4,
        linewidth=LINEWIDTH,
        alpha=ALPHA,
    )

    ax[1][1].loglog(
        times,
        Ns,
        color=color,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=4,
        linewidth=LINEWIDTH,
        alpha=ALPHA,
    )

    ax[1][2].loglog(
        Ns,
        chi2s,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=4,
        color=color,
        linewidth=LINEWIDTH,
        alpha=ALPHA,
    )


ax[1][0].loglog(
    tols,
    scipy_errors,
    color="darkgray",
    marker="o",
    markersize=4,
    linewidth=1.0,
    alpha=0.8,
    label="SciPy"
)
ax[0][0].loglog(
    scipy_Ns,
    scipy_errors,
    color="darkgray",
    marker="o",
    markersize=4,
    linewidth=1.0,
    alpha=0.8,
    label="SciPy"
)
ax[0][1].loglog(
    scipy_hs,
    scipy_errors,
    color="darkgray",
    marker="o",
    markersize=4,
    linewidth=1.0,
    alpha=0.8,
    label="SciPy"
)
ax[0][2].loglog(
    scipy_times,
    scipy_errors,
    color="darkgray",
    marker="o",
    markersize=4,
    linewidth=1.0,
    alpha=0.8,
    label="SciPy"
)
ax[1][1].loglog(
    scipy_times,
    scipy_Ns,
    color="darkgray",
    marker="o",
    markersize=4,
    linewidth=1.0,
    alpha=0.9,
    label="SciPy"
)

ax[0][0].set_xlabel("No. of grid points")
ax[0][0].set_ylabel("RMSE")

ax[0][1].set_xlabel("Largest step")
ax[0][1].set_ylabel("RMSE")

ax[0][2].set_xlabel("Runtime (s)")
ax[0][2].set_ylabel("RMSE")

ax[1][0].set_xlabel("Tolerance")
ax[1][0].set_ylabel("RMSE")

ax[1][1].set_xlabel("Runtime (s)")
ax[1][1].set_ylabel("No. of grid points")

ax[1][2].set_xlabel("No. of grid points")
ax[1][2].set_ylabel("$\\chi^2$ statistic")

# ax[0].set_xlabel(r"Tolerance")
# ax[1].set_xlabel(r"Runtime (s)")
# ax[2].set_xlabel("Runtime (s)")
# ax[3].set_xlabel(r"No. of grid points")
#
# ax[0].set_ylabel("RMSE")
# ax[1].set_ylabel("RMSE")
# ax[2].set_ylabel(r"No. of grid points")
# ax[3].set_ylabel("ANEES")

ax[1][2].axhspan(out[0], out[1], alpha=0.1, color="black", linewidth=0.0)
ax[1][2].axhline(1.0, color="black", linewidth=0.5)

ax[0][2].legend(
    fancybox=False, edgecolor="black", fontsize="x-small"
).get_frame().set_linewidth(0.5)
# ax[1].legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)
# ax[2].legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)

plt.savefig("./figures/problem7_workprecision.pdf")
plt.show()
