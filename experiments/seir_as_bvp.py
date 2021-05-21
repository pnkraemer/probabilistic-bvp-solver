import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.integrate import solve_bvp, solve_ivp

from bvps.problem_examples import seir, seir_as_bvp

ivp = seir()

sol = solve_ivp(ivp.f, (ivp.t0, ivp.tmax), ivp.y0, dense_output=True)

# plt.plot(sol.t, sol.y.T)
# plt.show()


final_value = sol.y[:, -1]  # + 5 * scipy.stats.norm.rvs(size=4, random_state=1)


bvp = seir_as_bvp(IRmax=final_value[[2, 3]], population_count=100)


initial_grid = np.linspace(bvp.t0, bvp.tmax, 150)
initial_guess = -100 * np.ones((4, len(initial_grid)))
refsol = solve_bvp(bvp.f, bvp.scipy_bc, initial_grid, initial_guess, tol=1e-12)


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

plt.plot(refsol.x, refsol.y[0], label="S", color="C0", linestyle="-", linewidth=1)
plt.plot(refsol.x, refsol.y[1], label="E", color="C1", linestyle="--", linewidth=1)
plt.plot(refsol.x, refsol.y[2], label="I", color="C2", linestyle=":", linewidth=1)
plt.plot(refsol.x, refsol.y[3], label="R", color="C3", linestyle="-.", linewidth=1)

# plt.plot(sol.t, sol.y[0], alpha=0.2, linewidth=5, color="C0")
# plt.plot(sol.t, sol.y[1], alpha=0.2, linewidth=5, color="C1")
# plt.plot(sol.t, sol.y[2], alpha=0.2, linewidth=5, color="C2")
# plt.plot(sol.t, sol.y[3], alpha=0.2, linewidth=5, color="C3")

plt.plot(bvp.t0, bvp.y0[0], "o", markersize=5, color="C2")
plt.plot(bvp.t0, bvp.y0[1], "d", markersize=5, color="C3")
plt.plot(bvp.tmax, bvp.ymax[0], "o", markersize=5, color="C2")
plt.plot(bvp.tmax, bvp.ymax[1], "d", markersize=5, color="C3")
plt.legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)

# ax.set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")

# plt.xlim((-5, 60))
# plt.ylim((0, 110))
plt.xticks((0.0, 50))
plt.yticks((0.0, 100.0))
plt.xlabel(r"Time")
plt.ylabel(r"Case counts")

plt.savefig("./figures/seir_as_bvp.pdf")
plt.show()
