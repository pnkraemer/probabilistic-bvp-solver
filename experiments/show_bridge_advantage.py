"""This script chisels out the advantage of using a bridge prior over not using one."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest
from probnum import filtsmooth, randvars, statespace
from tqdm import tqdm

from bvps import bvp_solver, problem_examples, quadrature

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
        "./visualization/stylesheets/fontsize/7pt.mplstyle",
        "./visualization/stylesheets/figsize/neurips/13_tile.mplstyle",
        "./visualization/stylesheets/misc/thin_lines.mplstyle",
        "./visualization/stylesheets/misc/bottomleftaxes.mplstyle",
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

for ax in ax_:
    ax.spines["left"].set_position(("outward", 2))
    ax.spines["bottom"].set_position(("outward", 2))

ax = ax_.reshape((2, 2))
colormaps = [
    plt.cm.Purples,
    plt.cm.Greens,
    plt.cm.Reds,
    plt.cm.Blues,
]
colormap_index_generator = itertools.count()
colormap_index = next(colormap_index_generator)
for initial_guess, row_axis in zip([2 * np.ones((N, bvp.dimension)), None], ax):
    for USE_BRIDGE, axis in zip([False, True], row_axis):
        cmap = colormaps[colormap_index]
        colormap_index = next(colormap_index_generator)
        ibm = statespace.IBM(
            ordint=4,
            spatialdim=bvp.dimension,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )

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
            tqdm(range(MAXIT)),
            itertools.chain([(initial_posterior, sigma_squared)], solution_gen),
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
            color = cmap(0.3 + 0.6 * i / MAXIT)
            alpha = 0.99
            linewidth = 2
            markersize = 5
            zorder = 1
            marker = "o"

            axis.plot(
                t,
                y[:, q],
                "-",
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                zorder=zorder,
            )
            axis.plot(
                kalman_posterior.locations,
                kalman_posterior.states.mean[:, 0],
                marker=marker,
                color=color,
                linestyle="None",
                alpha=alpha,
                linewidth=linewidth,
                markersize=markersize,
                zorder=zorder,
                markeredgewidth=0.5,
                markeredgecolor="black",
            )
            axis.plot(t, bvp.solution(t), color="black", linestyle="dashed")

            axis.set_xlim((-0.03, 1.03))
            axis.set_xticks((0.0, 0.5, 1.0))

            axis.set_ylim((0.5, 2.5))
            axis.set_yticks((0.5, 1.5, 2.5))

            if USE_BRIDGE:
                bridge_title = "Bridge"
            else:
                bridge_title = "Conventional"
            if initial_guess is None:
                EKS_title = "EKS"
            else:
                EKS_title = "Guess"
            axis.set_title(bridge_title + " & " + EKS_title, fontsize="medium")
            axis.set_xlabel("Time")

        # for q, curr_ax in zip(range(ibm.ordint + 1), axis):
        #     if q == 0:
        #         curr_ax.plot(t, bvp.solution(t), color="gray", linestyle="dashed")
        #         curr_ax.plot(t, y[:, q], color="k", alpha=0.1 + 0.9 * float(i / (MAXIT)))

        # for x in kalman_posterior.locations:
        #    curr_ax.axvline(x, linewidth=0.25, color="k")

ax[0][0].set_ylabel("Solution")
ax[1][1].set_ylim((0.5, 2.5))
plt.savefig("bridge_advantage.pdf")
plt.show()
