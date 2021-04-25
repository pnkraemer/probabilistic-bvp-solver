from bvps.problem_examples import seir, seir_as_bvp


from scipy.integrate import solve_ivp, solve_bvp

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

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
        "./visualization/stylesheets/science.mplstyle",
        "./visualization/stylesheets/misc/grid.mplstyle",
        # "./visualization/stylesheets/color/high-contrast.mplstyle",
        "./visualization/stylesheets/9pt.mplstyle",
        "./visualization/stylesheets/one_of_13_tile.mplstyle",
        # "./visualization/stylesheets/hollow_markers.mplstyle",
        "./visualization/stylesheets/probnum_colors.mplstyle",
    ]
)


fig, ax = plt.subplots(ncols=1, constrained_layout=True)

plt.plot(refsol.x, refsol.y[0], label="S", color="C0", linestyle="-")
plt.plot(refsol.x, refsol.y[1], label="E", color="C1", linestyle="--")
plt.plot(refsol.x, refsol.y[2], label="I", color="C2", linestyle=":")
plt.plot(refsol.x, refsol.y[3], label="R", color="C3", linestyle="-.")

plt.plot(sol.t, sol.y[0], alpha=0.2, linewidth=5, color="C0")
plt.plot(sol.t, sol.y[1], alpha=0.2, linewidth=5, color="C1")
plt.plot(sol.t, sol.y[2], alpha=0.2, linewidth=5, color="C2")
plt.plot(sol.t, sol.y[3], alpha=0.2, linewidth=5, color="C3")

plt.plot(bvp.t0, bvp.y0[0], "o", markersize=5, color="C2")
plt.plot(bvp.t0, bvp.y0[1], "d", markersize=5, color="C3")
plt.plot(bvp.tmax, bvp.ymax[0], "o", markersize=5, color="C2")
plt.plot(bvp.tmax, bvp.ymax[1], "d", markersize=5, color="C3")
plt.legend(fancybox=False, edgecolor="black").get_frame().set_linewidth(0.5)

ax.set_title(r"$\bf B$" + "  ", loc="left", fontweight="bold", ha="right")

# plt.xlim((-5, 60))
# plt.ylim((0, 110))
plt.xlabel(r"Time $t$")
plt.ylabel(r"Case counts")

plt.savefig("./figures/seir_as_bvp.pdf")
plt.show()
