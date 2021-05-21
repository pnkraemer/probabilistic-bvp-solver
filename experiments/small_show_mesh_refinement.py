"""Try out probsolve_bvp."""
import itertools
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from probnum import diffeq, filtsmooth
from probnum import random_variables as randvars
from probnum import randvars, statespace
from probnum._randomvariablelist import _RandomVariableList
from tqdm import tqdm

from bvps import bvp_solver, ode_measmods, problem_examples

# bvp = problem_examples.problem_24_second_order(xi=1e-2)
bvp = problem_examples.problem_7_second_order(xi=1e-3)
ibm = statespace.IBM(
    ordint=4,
    spatialdim=bvp.dimension,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
measmod = ode_measmods.from_second_order_ode(bvp, ibm)


MAXIT = 100
TOL = 1e-6

# Plotting parameters
t = np.linspace(bvp.t0, bvp.tmax, 150)
titles = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOG_YLIM = (1e-16, 1e4)

# plt.style.use(
#     [
#         # "./visualization/stylesheets/science.mplstyle",
#         # "./visualization/stylesheets/misc/grid.mplstyle",
#         "./visualization/stylesheets/color/high-contrast.mplstyle",
#         "./visualization/stylesheets/8pt.mplstyle",
#         "./visualization/stylesheets/23_tile_jmlr.mplstyle",
#         "./visualization/stylesheets/thin_lines.mplstyle",
#     ]
# )
# mpl.rcParams['lines.linewidth'] = 0.5
# mpl.rcParams['xtick.major.size'] = 2
# mpl.rcParams['xtick.minor.size'] = 2
# mpl.rcParams['ytick.major.size'] = 2
# mpl.rcParams['ytick.minor.size'] = 2
# mpl.rcParams['xtick.major.width'] = 0.5
# mpl.rcParams['xtick.minor.width'] = 0.5
# mpl.rcParams['ytick.major.width'] = 0.5
# mpl.rcParams['ytick.minor.width'] = 0.5


COLOR = "darkgreen"
SECOND_COLOR = "darkblue"


# Reference solution

initial_grid = np.linspace(bvp.t0, bvp.tmax, 1024, endpoint=True)
initial_guess = np.ones((len(initial_grid), bvp.dimension))


ibm = statespace.IBM(
    ordint=4,
    spatialdim=bvp.dimension,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
solver = bvp_solver.BVPSolver.from_default_values(ibm, initial_sigma_squared=1e2)

initial_posterior, _ = solver.compute_initialisation(
    bvp, initial_grid, initial_guess=None, use_bridge=True
)

MAXIT_IEKS = 1
MAXIT_EM = 20
solution_gen = solver.solution_generator(
    bvp,
    atol=TOL,
    rtol=TOL,
    initial_posterior=initial_posterior,
    maxit_ieks=MAXIT_IEKS,
    maxit_em=MAXIT_EM,
    yield_ieks_iterations=False,
)

# Skip initialisation
# next(solution_gen)

# First iteration
reference_posterior, sigma_squared = next(solution_gen)
# Evaluate the residual
residual_evaluations_at_locations = _RandomVariableList(
    [
        measmod.forward_rv(rv, t_)[0]
        for rv, t_ in zip(reference_posterior.states, reference_posterior.locations)
    ]
)
residual_mean_at_locations = residual_evaluations_at_locations.mean
residual_std_at_locations = residual_evaluations_at_locations.std * np.sqrt(
    sigma_squared
)
print(np.linalg.norm(residual_mean_at_locations))
print(np.linalg.norm(residual_std_at_locations))


plt.style.use(
    [
        "./visualization/stylesheets/fontsize/7pt.mplstyle",
        "./visualization/stylesheets/figsize/neurips/23_tile.mplstyle",
        "./visualization/stylesheets/misc/thin_lines.mplstyle",
        "./visualization/stylesheets/misc/bottomleftaxes.mplstyle",
    ]
)

# Set up all 7 subplots
fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    sharex="col",
    sharey="row",
    dpi=200,
    constrained_layout=True,
    gridspec_kw={"height_ratios": [1, 1]},
)


for ax in axes.flatten():
    ax.spines["left"].set_position(("outward", 2))
    ax.spines["bottom"].set_position(("outward", 2))

for axis_index, (N, ax) in enumerate(zip([5 ** 1, 5 ** 2, 5 ** 3, 5 ** 4], axes.T)):
    print(N)
    ibm = statespace.IBM(
        ordint=4,
        spatialdim=bvp.dimension,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
    solver = bvp_solver.BVPSolver.from_default_values(ibm, initial_sigma_squared=1e2)

    # ax[0].set_title(f"$N={N}$")
    initial_grid = np.linspace(bvp.t0, bvp.tmax, N, endpoint=True)
    initial_guess = np.ones((len(initial_grid), bvp.dimension))
    initial_posterior, sigma_squared = solver.compute_initialisation(
        bvp, initial_grid, initial_guess=None, use_bridge=True
    )

    solution_gen = solver.solution_generator(
        bvp,
        atol=TOL,
        rtol=TOL,
        initial_posterior=initial_posterior,
        maxit_ieks=MAXIT_IEKS,
        maxit_em=MAXIT_EM,
        yield_ieks_iterations=False,
    )

    # First iteration
    kalman_posterior, sigma_squared = next(solution_gen)
    posterior_evaluations = kalman_posterior(t)
    posterior_mean = posterior_evaluations.mean
    posterior_std = posterior_evaluations.std * np.sqrt(sigma_squared)

    idx = itertools.count()
    i = next(idx)

    # Evaluate the residual
    residual_evaluations_at_locations = _RandomVariableList(
        [
            measmod.forward_rv(rv, t_)[0]
            for rv, t_ in zip(kalman_posterior.states, kalman_posterior.locations)
        ]
    )
    residual_mean_at_locations = residual_evaluations_at_locations.mean
    residual_std_at_locations = residual_evaluations_at_locations.std * np.sqrt(
        sigma_squared
    )
    print(np.linalg.norm(residual_mean_at_locations))
    print(np.linalg.norm(residual_std_at_locations))

    # Evaluate the residual
    residual_evaluations = _RandomVariableList(
        [measmod.forward_rv(rv, t_)[0] for rv, t_ in zip(posterior_evaluations, t)]
    )
    residual_mean = residual_evaluations.mean
    residual_std = residual_evaluations.std * np.sqrt(sigma_squared)

    # Evaluate the reference solution and errors
    reference_evaluated = reference_posterior(t).mean[:, 0]
    SOL_YLIM = (-3, 3)
    reference_color = "black"
    reference_linestyle = "dotted"
    true_error = np.abs(reference_evaluated - posterior_mean[:, 0])

    # Extract plotting residuals and uncertainties
    error_estimates_std = np.abs(posterior_std[:, 0])
    error_estimates_res_mean = np.abs(residual_mean)
    error_estimates_res_std = np.abs(residual_std)

    #
    # for x in kalman_posterior.locations:
    #     ax[i].axvline(x, color=COLOR, linewidth=0.25)
    # ax[i].set_yticks([])
    #
    # i = next(idx)

    # First row: uncertainties derived from std
    assert t.shape == error_estimates_std.shape
    ax[i].semilogy(t, error_estimates_std, "-", color="black")

    if axis_index in [0, 1]:
        ax[i].fill_between(
            t, 1e-16 * np.ones_like(t), error_estimates_std, color=COLOR, alpha=0.5
        )
    else:
        ax[i].fill_between(
            t, 1e-16 * np.ones_like(t), error_estimates_std, color=COLOR, alpha=0.2
        )
    if axis_index == 0:
        ax[i].set_ylabel("Std. deviation")
    ax[i].semilogy(
        t,
        true_error,
        color="black",
        linestyle=reference_linestyle,
    )
    ax[i].set_ylim(LOG_YLIM)
    ax[i].set_yticks([LOG_YLIM[0], 1e-6, LOG_YLIM[1]])
    # ax[i].set_title(f"{titles[i]}{str(N)}", loc="left", fontweight="bold")
    ax[i].set_title(f"$N = {str(N)}$", fontsize="medium")

    i = next(idx)

    # Next row: Solution and uncertainties
    # ax[i].plot(
    #     t,
    #     reference_evaluated,
    #     color="black",
    #     linestyle=reference_linestyle,
    # )
    # ax[i].plot(t, posterior_mean[:, 0], color="black")
    # ax[i].fill_between(
    #     t,
    #     posterior_mean[:, 0] - 3 * posterior_std[:, 0],
    #     posterior_mean[:, 0] + 3 * posterior_std[:, 0],
    #     alpha=0.2,
    #     color=COLOR,
    #     linewidth=0,
    # )
    # ax[i].plot(t, posterior_mean[:, 1], color="black", linestyle="dashed")
    # ax[i].fill_between(
    #     t,
    #     posterior_mean[:, 1] - 3 * posterior_std[:, 1],
    #     posterior_mean[:, 1] + 3 * posterior_std[:, 1],
    #     alpha=0.2,
    #     color=COLOR,
    #     linewidth=0,
    # )
    # if axis_index == 0:
    #     ax[i].set_ylabel("Solution")
    # ax[i].set_ylim((-3, 5))
    # ax[i].set_yticks([-3, 5])
    # ax[i].set_title(f"{titles[i]}{str(N)}", loc="left", fontweight="bold")
    # i = next(idx)
    #
    # # Next row: Residual and uncertainties
    # ax[i].plot(
    #     t,
    #     residual_mean[:, 0],
    #     color="black",
    # )
    # ax[i].plot(
    #     t,
    #     np.zeros_like(t),
    #     color="black",
    #     linestyle=reference_linestyle,
    # )
    # ax[i].fill_between(
    #     t,
    #     residual_mean[:, 0] - 3 * residual_std[:, 0],
    #     residual_mean[:, 0] + 3 * residual_std[:, 0],
    #     alpha=0.2,
    #     color=SECOND_COLOR,
    #     linewidth=0,
    # )
    # if axis_index == 0:
    #     ax[i].set_ylabel("Residual")
    # ax[i].set_ylim((-2000, 2000))
    # ax[i].set_yticks([-2000, 2000])
    #
    # ax[i].set_title(f"{titles[i]}{str(N)}", loc="left", fontweight="bold")
    #
    # i = next(idx)

    # Next row: residual and uncertainties
    assert t.shape == error_estimates_res_mean[:, 0].shape
    ax[i].semilogy(
        t,
        error_estimates_res_mean[:, 0],
        color="black",
    )

    ax[i].semilogy(
        t,
        np.sqrt(
            error_estimates_res_mean[:, 0] ** 2 + error_estimates_res_std[:, 0] ** 2
        ),
        color="black",
        linestyle="dashed",
    )

    if axis_index in [2, 3]:
        ax[i].fill_between(
            t,
            1e-16 * np.ones_like(t),
            error_estimates_res_mean[:, 0],
            color=SECOND_COLOR,
            alpha=0.5,
        )
        ax[i].fill_between(
            t,
            error_estimates_res_mean[:, 0],
            np.sqrt(
                error_estimates_res_mean[:, 0] ** 2 + error_estimates_res_std[:, 0] ** 2
            ),
            color="black",
            alpha=0.2,
        )
    else:
        ax[i].fill_between(
            t,
            1e-16 * np.ones_like(t),
            error_estimates_res_mean[:, 0],
            color=SECOND_COLOR,
            alpha=0.2,
        )
        ax[i].fill_between(
            t,
            error_estimates_res_mean[:, 0],
            np.sqrt(
                error_estimates_res_mean[:, 0] ** 2 + error_estimates_res_std[:, 0] ** 2
            ),
            color="black",
            alpha=0.2,
        )

    ax[i].semilogy(
        t,
        true_error,
        color="black",
        linestyle=reference_linestyle,
    )
    ax[i].set_ylim(LOG_YLIM)
    ax[i].set_yticks([LOG_YLIM[0], 1e-6, LOG_YLIM[1]])
    if axis_index == 0:
        ax[i].set_ylabel("(Prob.) Residual")
    # ax[i].set_title(f"{titles[i]}{str(N)}", loc="left", fontweight="bold")
    i = next(idx)

    # Clean up: remove all ticks for now and show the plot
    for a in ax:
        a.set_xticks([bvp.t0, bvp.t0 + 0.5 * (bvp.tmax - bvp.t0), bvp.tmax])
        # a.set_yticks([])
        a.set_xlim((bvp.t0, bvp.tmax))

for ax in axes[-1]:
    ax.set_xlabel("Time")
fig.align_ylabels()
plt.savefig(f"errorestimates{N}.pdf")
plt.show()
