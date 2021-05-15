"""Plot WP on bratu's problem."""
import numpy as np
import json
import matplotlib.pyplot as plt
from probnumeval.timeseries import chi2_confidence_intervals

out = chi2_confidence_intervals(dim=1)

with open("./data/bratus_problem_work_precision.json", "r") as infile:
    data = json.load(infile)
plt.style.use(
    [
        "./visualization/stylesheets/science.mplstyle",
        "./visualization/stylesheets/misc/grid.mplstyle",
        # "./visualization/stylesheets/tufte_colors.mplstyle",
        "./visualization/stylesheets/13_tile_jmlr.mplstyle",
        "./visualization/stylesheets/9pt.mplstyle",
        "./visualization/stylesheets/probnum_colors.mplstyle",
    ]
)


fig, ax = plt.subplots(ncols=4, dpi=150, constrained_layout=True)
markers = ["s", "d", "^", "o"]
colors = ["C0", "C1", "C2", "C3"]
for q, marker, color in zip(data.keys(), markers, colors):
    qres = data[q]
    Ns = []
    chi2s = []
    errors = []
    tols = []
    times = []
    for tol in qres.keys():
        qtolres = qres[tol]

        chi2 = qtolres["chi2"]
        error = qtolres["error"]
        N = qtolres["N"]
        t = qtolres["time"]

        tols.append(float(tol))
        Ns.append(N)
        chi2s.append(chi2)
        errors.append(error)
        times.append(t)

    ax[0].loglog(
        tols,
        np.array(tols),
        color="gray",
        linestyle="dotted",
    )
    ax[0].loglog(
        tols,
        errors,
        color=color,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=4,
    )
    ax[1].loglog(
        Ns,
        errors,
        color=color,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=4,
    )
    ax[2].loglog(
        Ns,
        chi2s,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=4,
        color=color,
    )
    ax[3].loglog(
        Ns,
        times,
        label=f"$\\nu={q}$",
        marker=marker,
        markersize=4,
        color=color,
    )


ax[0].set_xlabel(r"Tolerance")
ax[1].set_xlabel(r"Number of grid points")
ax[2].set_xlabel(r"Number of grid points")
ax[3].set_xlabel(r"Number of grid points")

ax[0].set_ylabel("RMSE")
ax[1].set_ylabel("RMSE")
ax[2].set_ylabel("ANEES")
ax[3].set_ylabel("Time (s)")

ax[2].axhspan(out[0], out[1], alpha=0.1, color="black", linewidth=0.0)
ax[2].axhline(1.0, color="black", linewidth=0.5)

ax[0].legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)
ax[1].legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)
ax[2].legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)

plt.savefig("./figures/bratu_workprecision.pdf")
plt.show()
