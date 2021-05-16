from bvps import problem_examples, bvp_solver, ode_measmods
from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import itertools

from probnum import diffeq, statespace

bvp = problem_examples.problem_20_second_order(xi=0.00125)

ibm = statespace.IBM(
    ordint=7,
    spatialdim=bvp.dimension,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
measmod = ode_measmods.from_second_order_ode(bvp, ibm)
MAXIT = 25


plt.style.use(
    [
        "./visualization/stylesheets/fontsize/7pt.mplstyle",
        "./visualization/stylesheets/figsize/neurips/23_tile.mplstyle",
        "./visualization/stylesheets/misc/thin_lines.mplstyle",
        "./visualization/stylesheets/misc/bottomleftaxes.mplstyle",
        "./visualization/stylesheets/marker/framed_markers.mplstyle",
    ]
)

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    dpi=200,
    constrained_layout=True,
    sharex="col",
    sharey="row",
    gridspec_kw={"height_ratios": [2, 5]},
)

for ax in axes.flatten():
    ax.spines["left"].set_position(("outward", 2))
    ax.spines["bottom"].set_position(("outward", 2))


initial_grid = np.linspace(bvp.t0, bvp.tmax, 6, endpoint=True)
t = np.linspace(bvp.t0, bvp.tmax, 50, endpoint=True)

for MAXIT_IEKS, axis_col, colormap, marker in zip(
    [MAXIT, 5, 1], axes.T, [plt.cm.Blues, plt.cm.Oranges, plt.cm.Purples], ["o", "^", "d"]
):
    # Solver and solver parameters

    solver = bvp_solver.BVPSolver.from_default_values(ibm, initial_sigma_squared=1e2)
    initial_posterior, sigma_squared = solver.compute_initialisation(
        bvp, initial_grid, initial_guess=None, use_bridge=True
    )

    solution_gen = solver.solution_generator(
        bvp,
        atol=1e-0,
        rtol=1e-0,
        initial_posterior=initial_posterior,
        maxit_ieks=MAXIT_IEKS,
        maxit_em=MAXIT // MAXIT_IEKS,
        yield_ieks_iterations=True,
    )

    # Only the first 20 iterations, i.e. the first 4 refinements (since maxit_ieks=5)
    for iteration, (reference_posterior, sigma_squared) in zip(
        tqdm(range(MAXIT)),
        itertools.chain([(initial_posterior, sigma_squared)], solution_gen),
    ):

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

        # print(np.mean(y[:, 0]), len(reference_posterior.locations))
        # for t in x:
        #     plt.axvline(t, linewidth=0.5, color="k")
        # plt.plot(x, y[:, 0])
        # plt.plot(x, y[:, 1:3])
        axis_col[0].plot(t, bvp.solution(t), color="black", linestyle="dotted")
        axis_col[0].plot(
            t, y2[:, 0], color=colormap(0.2 + 0.6* iteration / MAXIT), alpha=0.9
        )
        axis_col[0].plot(
            x,
            y[:, 0],
            marker=marker,
            color=colormap(0.2 + 0.7*iteration / MAXIT),
            alpha=0.9,
            markersize=5,
        )
        axis_col[1].semilogy(
            x,
            (np.abs(res) + 1e-18),
            marker=marker,
            linestyle="None",
            color=colormap(0.2 + 0.7*iteration / MAXIT),
            nonpositive="clip",
            alpha=0.9,
            markersize=5,
        )
        if iteration == MAXIT - 1:
            axis_col[1].semilogy(
                x,
                np.abs(res) + 1e-18,
                marker=marker,
                linestyle="-",
                linewidth=2,
                color=colormap(0.2 + 0.7 * iteration / MAXIT),
                nonpositive="clip",
                alpha=0.5,
                markersize=5,
            )
        # axis_col[1].semilogy(t, np.abs(res2),color="gray", nonpositive="clip")
        # axis_col[0].set_ylim((-0.02, 0.02))
        axis_col[1].set_ylim((1e-21, 1e1))
        axis_col[0].set_title(
            f"{MAXIT_IEKS} IEKS Updates, {MAXIT // MAXIT_IEKS} EM Updates",
            fontsize="medium",
        )
        axis_col[1].set_xlim((-0.03, 1.03))
        axis_col[1].set_xticks((0.0, 0.5, 1.0))
        axis_col[0].set_ylim((0.5, 2.5))
        axis_col[0].set_yticks((0.5, 2.5))
        axis_col[1].set_ylim((1e-14, 1e2))
        axis_col[1].set_yticks((1e-14, 1e-6, 1e2))

# colbar = fig.colorbar(ax=axes.ravel().tolist(), pad=0.04, aspect = 30)

axes[0][0].set_ylabel("Solution")
axes[1][0].set_ylabel("Residual")
axes[-1][0].set_xlabel("Time")
axes[-1][1].set_xlabel("Time")
axes[-1][2].set_xlabel("Time")

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
