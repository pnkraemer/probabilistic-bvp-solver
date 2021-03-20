"""Template to turn the .csv files in data/ into work-precision plots."""
import pandas as pd

import matplotlib.pyplot as plt
results = pd.read_csv("data/workprecision_first_attempt_bratus.csv", index_col=0)


for colidx in results.columns:
    plt.loglog(results.index, results[colidx], marker="o", label=colidx)

plt.xlabel("N")
plt.ylabel("RMSE")
plt.legend()
plt.show()
print(results.columns)
print(results.index)
