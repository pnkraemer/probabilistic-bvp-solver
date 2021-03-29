"""Template to turn the .csv files in data/ into work-precision plots."""
import pandas as pd

import matplotlib.pyplot as plt
from probnumeval.timeseries import chi2_confidence_intervals
from _styles import LINESTYLES, MARKERS

out = chi2_confidence_intervals(dim=2)
print(out)
results_rmse = pd.read_csv(
    "data/workprecision_first_attempt_r_example_rmse.csv", index_col=0
)
results_anees = pd.read_csv(
    "data/workprecision_first_attempt_r_example_anees.csv", index_col=0
)
results_nci = pd.read_csv(
    "data/workprecision_first_attempt_r_example_nci.csv", index_col=0
)


results_rmse2 = pd.read_csv(
    "data/workprecision_first_attempt_r_example_rmse_q3.csv", index_col=0
)
results_anees2 = pd.read_csv(
    "data/workprecision_first_attempt_r_example_anees_q3.csv", index_col=0
)
results_nci2 = pd.read_csv(
    "data/workprecision_first_attempt_r_example_nci_q3.csv", index_col=0
)


plt.style.use(
    [
        "./visualization/science.mplstyle",
        "./visualization/misc/grid.mplstyle",
        "./visualization/color/high-contrast.mplstyle",
        "./stylesheets/13_tile_jmlr.mplstyle",
        "./stylesheets/10pt.mplstyle",
        # "./stylesheets/probnum_colors.mplstyle"
    ]
)

fig, ax = plt.subplots(ncols=2, dpi=150, constrained_layout=True)


for colidx, linestyle, marker in zip(results_rmse.columns[:1], LINESTYLES, MARKERS):
    ax[0].loglog(
        results_rmse.index,
        results_rmse[colidx],
        label=r"$\nu=4$",
        linestyle=linestyle,
        marker=marker,
    )
    ax[1].loglog(
        results_anees.index,
        results_anees[colidx],
        linestyle=linestyle,
        marker=marker,
        label=r"$\nu=4$",
    )
    # ax[2].semilogx(results_nci.index, results_nci[colidx], marker="o", label="q=4")

    ax[0].loglog(
        results_rmse2.index, results_rmse2[colidx], marker=marker, label=r"$\nu=3$"
    )
    ax[1].loglog(
        results_anees2.index, results_anees2[colidx], marker=marker, label=r"$\nu=3$"
    )
    # ax[2].semilogx(results_nci2.index, results_nci2[colidx], marker="o", label="q=3")
ax[0].grid(which="minor")
ax[1].grid(which="minor")


ax[1].axhspan(out[0], out[1], alpha=0.1, color="black", linewidth=0.0)
ax[1].axhline(1.0, color="black", linewidth=0.5)
# ax[1].fill_between(
#     results_anees.index, out[0], out[1], color="green", alpha=0.25, label="99% Conf."
# )


for axis in ax:
    axis.set_xlabel(r"Mesh-size, $N$")
    axis.legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)


ax[0].set_ylabel(r"RMSE, $\varepsilon$")
ax[1].set_ylabel(r"ANEES, $\chi^2$")

ax[0].set_title(r"$\bf A$" + "  ", loc="left", fontweight="bold", ha="right")
ax[1].set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")

# ax[2].set_title("NCI")
plt.savefig("figures/r_example_results.pdf")
plt.show()
