"""This script chisels out the advantage of using a bridge prior over not using one."""

from bvps import bvp_solver, problem_examples, quadrature
import numpy as np

import pytest
from probnum import statespace, filtsmooth, randvars
import matplotlib.pyplot as plt
from tqdm import tqdm


import itertools
# bvp = problem_examples.problem_7_second_order(xi=1e-2)
bvp = problem_examples.problem_20_second_order(xi=1e-2)
ibm = statespace.IBM(
    ordint=4,
    spatialdim=bvp.dimension,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
N = 6
initial_grid = np.linspace(bvp.t0, bvp.tmax, N, endpoint=True)
t = np.linspace(bvp.t0, bvp.tmax, 100)

MAXIT = 5


plt.style.use(
    [
        "./visualization/stylesheets/science.mplstyle",
        # "./visualization/stylesheets/misc/grid.mplstyle",
        "./visualization/stylesheets/9pt.mplstyle",
        "./visualization/stylesheets/13_tile_jmlr.mplstyle",
        "./visualization/stylesheets/baby_colors.mplstyle",
    ]
)
fig, ax_ = plt.subplots(
    ncols=4,
    nrows=1,
    sharey=True,
    sharex=True,
    gridspec_kw={"height_ratios": [4]},
    constrained_layout=True,
)
ax = ax_.reshape((2, 2))
colormaps = [plt.cm.Blues, plt.cm.Greens,plt.cm.Reds,plt.cm.Purples,]
colormap_index_generator = itertools.count()
colormap_index = next(colormap_index_generator)
for initial_guess, row_axis in zip([None, 2*np.ones((N, bvp.dimension))], ax):
    for USE_BRIDGE, axis in zip(
        [True, False], row_axis
    ):
        cmap = colormaps[colormap_index]
        colormap_index = next(colormap_index_generator)

        solver = bvp_solver.BVPSolver.from_default_values(
            ibm, initial_sigma_squared=1e8
        )
        initial_posterior, sigma_squared = solver.compute_initialisation(
            bvp, initial_grid, initial_guess, use_bridge=USE_BRIDGE
        )

        solution_gen = solver.solution_generator(
            bvp,
            atol=1e-0,
            rtol=1e-0,
            initial_posterior=initial_posterior,
            maxit_ieks=MAXIT,
            yield_ieks_iterations=True,
        )

        # Only the first 20 iterations, i.e. the first 4 refinements (since maxit_ieks=5)
        for i, (kalman_posterior, sigma_squared) in zip(
            tqdm(range(MAXIT)), itertools.chain([(initial_posterior, sigma_squared)], solution_gen)
        ):

            y = kalman_posterior(t).mean
            s = kalman_posterior(t).std * np.sqrt(sigma_squared)
            # axis[0].plot(t, y[:, 0], color="k", alpha=0.3 + 0.7 * float(i / (MAXIT)))
            # for q, curr_ax in zip(range(ibm.ordint + 1), axis):
            #     curr_ax.plot(t, y[:, q], color="k", alpha=0.3 + 0.7 * float(i / (MAXIT)))
            #     # curr_ax.fill_between(t, y[:, q] - 2 * s[:, q], y[:, q] + 2*s[:, q], color="k", alpha=0.1)
            #     #
            #     curr_ax.plot(
            #         kalman_posterior.locations,
            #         kalman_posterior.states.mean[:, q],
            #         ".",
            #         color="black",
            #         alpha=0.3 + 0.7 * float(i / (MAXIT)),
            #     )
            q = 0
            color = cmap(0.3 + 0.6*i / MAXIT)
            alpha = 0.99
            linewidth = 2
            markersize = 5
            zorder=1
            marker="o"

            axis.plot(t, y[:, q], "-", color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
            axis.plot(
                kalman_posterior.locations,
                kalman_posterior.states.mean[:, 0],
                marker=marker,
                color=color,
                linestyle="None",
                alpha=alpha,
                linewidth=linewidth,
                markersize=markersize, zorder=zorder,
                markeredgewidth=0.5,
                markeredgecolor="black",
            )
            axis.plot(t, bvp.solution(t), color="black", linestyle="dashed")

            if USE_BRIDGE:
                bridge_title = "Bridge"
            else:
                bridge_title = "Conventional"
            if initial_guess is None:
                EKS_title = "EKS"
            else:
                EKS_title = "Guess"
            axis.set_title(bridge_title + r" $\&$ " + EKS_title)
            axis.set_xlabel("Time $t$")

        # for q, curr_ax in zip(range(ibm.ordint + 1), axis):
        #     if q == 0:
        #         curr_ax.plot(t, bvp.solution(t), color="gray", linestyle="dashed")
        #         curr_ax.plot(t, y[:, q], color="k", alpha=0.1 + 0.9 * float(i / (MAXIT)))

        # for x in kalman_posterior.locations:
        #    curr_ax.axvline(x, linewidth=0.25, color="k")

ax[0][0].set_ylabel("Solution $y$")
ax[1][1].set_ylim((0.5, 2.5))
plt.savefig("bridge_advantage.pdf")
plt.show()
