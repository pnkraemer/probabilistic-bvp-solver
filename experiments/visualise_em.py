from bvps import problem_examples, bvp_solver, ode_measmods
from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from probnum import diffeq, statespace

bvp = problem_examples.problem_20_second_order()

ibm = statespace.IBM(
    ordint=6,
    spatialdim=bvp.dimension,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
measmod = ode_measmods.from_second_order_ode(bvp, ibm)
MAXIT = 36


plt.style.use(
    [
        "./visualization/stylesheets/science.mplstyle",
        # "./visualization/stylesheets/misc/grid.mplstyle",
        "./visualization/stylesheets/8pt.mplstyle",
        "./visualization/stylesheets/13_tile_jmlr.mplstyle",
        "./visualization/stylesheets/baby_colors.mplstyle",
    ]
)

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    dpi=200,
    constrained_layout=True,
    sharex="col",
    sharey="row",
)

initial_grid = np.linspace(bvp.t0, bvp.tmax, 6, endpoint=True)
t = np.linspace(bvp.t0, bvp.tmax, 50, endpoint=True)

for MAXIT_IEKS, axis_col, color in zip([MAXIT, 6, 1], axes.T, ["C0", "C1", "C2"]):
    # Solver and solver parameters
    solver = bvp_solver.BVPSolver.from_default_values_std_refinement(
        ibm, use_bridge=True, initial_sigma_squared=1e5
    )
    TOL = 1e-3

    solution_gen = solver.solution_generator(
        bvp,
        atol=TOL,
        rtol=TOL,
        initial_grid=initial_grid,
        initial_guess=None,
        maxit_ieks=MAXIT_IEKS,
        maxit_em=MAXIT // MAXIT_IEKS,
        yield_ieks_iterations=True,
    )
    # Skip initialisation
    for iteration, (reference_posterior, _) in zip(range(MAXIT), solution_gen):

        posterior = diffeq.KalmanODESolution(reference_posterior)

        x = reference_posterior.locations
        y = reference_posterior(x).mean

        y2 = reference_posterior(t).mean

        res = np.array(
            [measmod.forward_realization(y_, t=x_)[0].mean for (x_, y_) in zip(x, y)]
        )
        res2 = np.array(
            [measmod.forward_realization(y_, t=x_)[0].mean for (x_, y_) in zip(t, y2)]
        )

        print(np.mean(y[:, 0]), len(reference_posterior.locations))
        # for t in x:
        #     plt.axvline(t, linewidth=0.5, color="k")
        # plt.plot(x, y[:, 0])
        # plt.plot(x, y[:, 1:3])
        axis_col[0].plot(t, bvp.solution(t), color="black", linestyle="dotted")
        axis_col[0].plot(t, y2[:, 0], color=plt.cm.cividis(1 - iteration / MAXIT))
        axis_col[0].plot(
            x,
            y[:, 0],
            "o",
            color=plt.cm.cividis(1 - iteration / MAXIT),
            markeredgecolor="black",
            markeredgewidth=0.01,
            alpha=0.95,
            markersize=4,
        )
        axis_col[1].semilogy(
            x,
            np.abs(res) + 1e-18,
            "o",
            color=plt.cm.cividis(1 - iteration / MAXIT),
            nonpositive="clip",
            markeredgecolor="black",
            markeredgewidth=0.01,
            alpha=0.95,
            markersize=4,
        )
        # axis_col[1].semilogy(t, np.abs(res2),color="gray", nonpositive="clip")
        # axis_col[0].set_ylim((-0.02, 0.02))
        axis_col[1].set_ylim((1e-21, 1e1))
        axis_col[0].set_title(
            f"$I_\\text{{IEKS}} = {MAXIT_IEKS}$, $I_\\text{{EM}} = {MAXIT // MAXIT_IEKS - 1}$"
        )

# colbar = fig.colorbar(ax=axes.ravel().tolist(), pad=0.04, aspect = 30)

axes[0][0].set_ylabel("Solution")
axes[1][0].set_ylabel("Residual")
axes[-1][0].set_xlabel("Time $t$")
axes[-1][1].set_xlabel("Time $t$")
axes[-1][2].set_xlabel("Time $t$")

#
# solver2 = bvp_solver.BVPSolver.from_default_values_std_refinement(
#     ibm, use_bridge=True, initial_sigma_squared=1e5
# )
# TOL = 1e-3
#
# solution_gen2 = solver2.solution_generator(
#     bvp,
#     atol=TOL,
#     rtol=TOL,
#     initial_grid=initial_grid,
#     initial_guess=None,
#     maxit_ieks=8,
#     maxit_em=1,
#     yield_ieks_iterations=True,
# )
# # Skip initialisation
# next(solution_gen2)
# for _, (reference_posterior2, _) in zip(range(8), solution_gen2):
#     print(".")
#     x = reference_posterior2.locations
#     y5 = reference_posterior2(x).mean
#
#     res4 = np.array(
#         [measmod.forward_realization(y_, t=x_)[0].mean for (x_, y_) in zip(x, y5)]
#     )
#
#     axes[0][0].plot(
#         x,
#         y5[:, 0],
#         "o",
#         markerfacecolor="white",
#         markeredgecolor="C0",
#         markeredgewidth=0.8,
#         alpha=0.5,
#     )
#     axes[1][0].semilogy(
#         x,
#         np.abs(res4) + 1e-18,
#         "o",
#         markerfacecolor="white",
#         markeredgecolor="C0",
#         markeredgewidth=0.8,
#         alpha=0.5,
#         nonpositive="clip",
#     )


# mpl.colorbar.ColorbarBase(ax=axes.ravel().tolist(), cmap=plt.cm.magma)


fig.align_ylabels()
plt.savefig("./figures/em_reasoning.pdf")
plt.show()
