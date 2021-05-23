"""Plot WP on bratu's problem."""
import json

import matplotlib.pyplot as plt
import numpy as np
from probnumeval.timeseries import chi2_confidence_intervals

out = chi2_confidence_intervals(dim=1, perc=0.95)

with open("./data/bratus_problem_work_precision.json", "r") as infile:
    data = json.load(infile)

plt.style.use(
    [
        "./visualization/stylesheets/fontsize/7pt.mplstyle",
        "./visualization/stylesheets/figsize/neurips/13_tile.mplstyle",
        "./visualization/stylesheets/misc/thin_lines.mplstyle",
        "./visualization/stylesheets/misc/bottomleftaxes.mplstyle",
        "./visualization/stylesheets/misc/no_minor_ticks.mplstyle",
        "./visualization/stylesheets/color/probnum_colors.mplstyle",
    ]
)


fig, ax = plt.subplots(
    ncols=4,
    nrows=1,
    gridspec_kw={"height_ratios": [4]},
    constrained_layout=True,
)

for axis in ax:
    axis.spines["left"].set_position(("outward", 2))
    axis.spines["bottom"].set_position(("outward", 2))

    # axis.tick_params(which='major')
    axis.minorticks_off()

markers = ["s", "d", "^", "o"]
colors = ["C0", "C1", "C2", "C3"]
for q, marker, color in zip(data.keys(), markers, colors):
    qres = data[q]
    tols = []
    chi2s = []
    Ns = []
    errors = []
    times = []

    scipy_Ns = []
    scipy_errors = []
    scipy_times = []
    for tol in qres.keys():
        qtolres = qres[tol]

        chi2 = qtolres["chi2"]
        error = qtolres["error"]
        N = qtolres["N"]
        t = qtolres["time"]

        scipy_error = qtolres["scipy_error"]
        scipy_N = qtolres["scipy_N"]
        scipy_t = qtolres["scipy_time"]

        tols.append(float(tol))
        chi2s.append(chi2)
        Ns.append(N)
        errors.append(error)
        times.append(t)
        scipy_Ns.append(scipy_N)
        scipy_errors.append(scipy_error)
        scipy_times.append(scipy_t)

    # ax[0].loglog(
    #     tols,
    #     np.array(tols),
    #     color="gray",
    #     linestyle="dotted",
    # )
    # ax[0].loglog(
    #     tols,
    #     errors,
    #     color=color,
    #     label=f"$\\nu={q}$",
    #     marker=marker,
    #     markersize=3,
    # )
    ax[0].loglog(
        Ns,
        errors,
        color=color,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=3,
        linewidth=1.0,
    )
    ax[1].loglog(
        times,
        errors,
        color=color,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=3,
        linewidth=1.0,
    )
    ax[2].loglog(
        Ns,
        times,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=3,
        color=color,
        linewidth=1.0,
    )

    ax[3].loglog(
        Ns,
        chi2s,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=3,
        color=color,
        linewidth=1.0,
    )


print(scipy_Ns, scipy_errors, scipy_times)


ax[0].loglog(
    scipy_Ns,
    scipy_errors,
    color="gray",
    marker="*",
    markersize=3,
    linewidth=3.0,
    alpha=0.3
)
ax[1].loglog(
    scipy_times,
    scipy_errors,
    color="gray",
    marker="*",
    markersize=3,
    linewidth=3.0,
    alpha=0.3
)
ax[2].loglog(
    scipy_Ns,
    scipy_times,
    label="SciPy",
    color="gray",
    marker="*",
    markersize=3,
    linewidth=3.0,
    alpha=0.3
)




ax[0].set_xlabel(r"No. of grid points")
ax[1].set_xlabel(r"Time (s)")
ax[2].set_xlabel(r"No. of grid points")
ax[3].set_xlabel(r"No. of grid points")

ax[0].set_ylabel("RMSE")
ax[1].set_ylabel("RMSE")
ax[2].set_ylabel("Time (s)")
ax[3].set_ylabel("ANEES")

#
# ax[0].set_ylim((1e-16, 1e-0))
# ax[1].set_ylim((1e-16, 1e-0))
ax[2].set_ylim((2e-3, 3e0))
# ax[2].set_yticks((1e-2, 1e0, 1e2))
# ax[3].set_ylim((1e-6, 1e4))
# ax[3].set_yticks((1e-6, 1e-1, 1e4))
#
# ax[0].set_xlim((1e0, 1e4))
# ax[1].set_xlim((1e-2, 1e2))
# ax[2].set_xlim((1e0, 1e4))
# ax[3].set_xlim((1e0, 1e4))


ax[3].axhspan(out[0], out[1], alpha=0.1, color="black", linewidth=0.0)
ax[3].axhline(1.0, color="black", linewidth=0.5)

ax[2].legend(fancybox=False, edgecolor="black", fontsize="small",handlelength=1., loc="center right").get_frame().set_linewidth(0.5)
# ax[1].legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)
# ax[2].legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)
# ax[3].legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)
plt.minorticks_off()
plt.savefig("./figures/bratu_workprecision.pdf")
plt.show()
