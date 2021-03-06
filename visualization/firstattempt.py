"""Template to turn the .csv files in data/ into work-precision plots."""
import matplotlib.pyplot as plt
import pandas as pd
from probnumeval.timeseries import chi2_confidence_intervals

out = chi2_confidence_intervals(dim=2)
print(out)
results_rmse = pd.read_csv(
    "data/workprecision_first_attempt_bratus_rmse.csv", index_col=0
)
results_anees = pd.read_csv(
    "data/workprecision_first_attempt_bratus_anees.csv", index_col=0
)
results_nci = pd.read_csv(
    "data/workprecision_first_attempt_bratus_nci.csv", index_col=0
)

fig, ax = plt.subplots(ncols=3, figsize=(12, 4), dpi=200, tight_layout=True)


for colidx in results_rmse.columns:
    ax[0].loglog(results_rmse.index, results_rmse[colidx], marker="o", label=colidx)
    ax[1].loglog(results_anees.index, results_anees[colidx], marker="o", label=colidx)
    ax[2].semilogx(results_nci.index, results_nci[colidx], marker="o", label=colidx)


ax[1].fill_between(
    results_anees.index, out[0], out[1], color="green", alpha=0.25, label="99% Conf."
)


for axis in ax:
    axis.set_xlabel("N")
    axis.legend()

ax[0].set_title("RMSE")
ax[1].set_title("ANEES")
ax[2].set_title("NCI")
plt.savefig("figures/bratus_results.pdf")
plt.show()
