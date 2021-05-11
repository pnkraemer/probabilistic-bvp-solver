"""Try out probsolve_bvp."""
import numpy as np
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars, filtsmooth, diffeq
from probnum._randomvariablelist import _RandomVariableList
from bvps import bvp_solver, problem_examples, ode_measmods
from tqdm import tqdm
import pandas as pd

import itertools
from probnum import random_variables as randvars


bvp = problem_examples.problem_24_second_order(xi=1e-2)
ibm = statespace.IBM(
    ordint=4,
    spatialdim=bvp.dimension,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
measmod = ode_measmods.from_second_order_ode(bvp, ibm)


# Solver and solver parameters
solver = bvp_solver.BVPSolver.from_default_values(
    ibm, use_bridge=True, initial_sigma_squared=1e2
)
MAXIT = 5
TOL = 1e-6

# Plotting parameters
t = np.linspace(bvp.t0, bvp.tmax, 200)
titles = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOG_YLIM = (1e-12, 1e4)
plt.style.use(
    [
        "./visualization/stylesheets/science.mplstyle",
        "./visualization/stylesheets/misc/grid.mplstyle",
        # "./visualization/stylesheets/color/high-contrast.mplstyle",
        "./visualization/stylesheets/8pt.mplstyle",
        "./visualization/stylesheets/one_of_13_tile.mplstyle",
        # "./visualization/stylesheets/hollow_markers.mplstyle",
        "./visualization/stylesheets/probnum_colors.mplstyle",
    ]
)

for N in [2**5, 2**7, 2**9]:

    initial_grid = np.linspace(bvp.t0, bvp.tmax, N)
    solution_gen = solver.solution_generator(
        bvp,
        atol=TOL,
        rtol=TOL,
        initial_grid=initial_grid,
        initial_guess=None,
        maxit_ieks=MAXIT,
        maxit_em=1,
        yield_ieks_iterations=False,
    )
    # Skip initialisation
    next(solution_gen)

    # First iteration
    kalman_posterior, sigma_squared = next(solution_gen)
    posterior_evaluations = kalman_posterior(t)
    posterior_mean = posterior_evaluations.mean
    posterior_std = posterior_evaluations.std * np.sqrt(sigma_squared)

    # Set up all 7 subplots
    fig, ax = plt.subplots(
        nrows=6,
        sharex=True,
        dpi=200,
        figsize=(2, 3),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [8, 8, 8, 8, 8, 8]},
    )
    idx = itertools.count()
    i = next(idx)

    # Evaluate the residual
    residual_evaluations = _RandomVariableList(
        [measmod.forward_rv(rv, t)[0] for rv, t in zip(posterior_evaluations, t)]
    )
    residual_mean = residual_evaluations.mean
    residual_std = residual_evaluations.std * np.sqrt(sigma_squared)

    # Evaluate the reference solution and errors
    reference_evaluated = kalman_posterior(t).mean[:, 0]
    SOL_YLIM = (1.5 * np.amin(reference_evaluated), np.amax(reference_evaluated) * 1.5)
    # SOL_YLIM = (0.2, 1.6)
    reference_color = "crimson"
    reference_linestyle = "dotted"
    true_error = np.abs(reference_evaluated - posterior_mean[:, 0])
    # true_error = np.zeros_like(t)

    # Extract plotting residuals and uncertainties
    error_estimates_std = np.abs(posterior_std[:, 0])
    error_estimates_res_mean = np.abs(residual_mean)
    error_estimates_res_std = np.abs(residual_std)

    # Select grids to be plotted
    grid_refine_via_std = kalman_posterior.locations  # dummy
    grid_refine_via_prob_res = kalman_posterior.locations  # dummy
    grid_refine_via_res = kalman_posterior.locations

    # # First row: std-refinement grid
    # for t in grid_refine_via_std:
    #     ax[i].axvline(t, color="black", linewidth=0.5)
    # ax[i].set_title(
    #     f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    # )
    # i = next(idx)

    # Second row: uncertainties derived from std
    ax[i].set_title(f"N: {len(kalman_posterior.locations)}")
    # ax[i].semilogy(t, error_estimates_std, ":", color="black")
    ax[i].semilogy(t, error_estimates_std ** 2, "-", color="black")
    ax[i].set_ylabel("$\\log \\mathbb{V}[Y_0](t)$", fontsize="x-small")
    ax[i].semilogy(
        t,
        true_error ** 2,
        color=reference_color,
        linestyle=reference_linestyle,
    )
    ax[i].axhline(TOL, color=reference_color)
    ax[i].set_title(
        f"$\\bf {titles[i]}$",
        loc="left",
        fontweight="bold",
        fontsize="x-small",
    )
    ax[i].set_ylim(LOG_YLIM)
    i = next(idx)

    # Third row: Solution and uncertainties
    ax[i].plot(
        t,
        reference_evaluated,
        color=reference_color,
        linestyle=reference_linestyle,
    )
    ax[i].plot(t, posterior_mean[:, 0], color="black")
    ax[i].fill_between(
        t,
        posterior_mean[:, 0] - 3 * posterior_std[:, 0],
        posterior_mean[:, 0] + 3 * posterior_std[:, 0],
        alpha=0.2,
        color="black",
        linewidth=0,
    )
    ax[i].set_ylabel("$Y_0(t)$", fontsize="x-small")
    ax[i].set_title(
        f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    )
    ax[i].set_ylim(SOL_YLIM)
    i = next(idx)

    # Fourth row: Residual and uncertainties
    ax[i].plot(
        t,
        residual_mean[:, 0],
        color="black",
    )
    ax[i].plot(
        t,
        np.zeros_like(t),
        color=reference_color,
        linestyle=reference_linestyle,
    )
    ax[i].fill_between(
        t,
        residual_mean[:, 0] - 3 * residual_std[:, 0],
        residual_mean[:, 0] + 3 * residual_std[:, 0],
        alpha=0.2,
        color="black",
        linewidth=0,
    )
    ax[i].set_ylabel("$\\delta(Y)(t)$", fontsize="x-small")
    ax[i].set_title(
        f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    )

    i = next(idx)

    # Fifth row: Log-residual and uncertainties
    ax[i].semilogy(
        t,
        error_estimates_res_mean[:, 0] ** 2,
        color="black",
    )
    # # ax[i].semilogy(
    # #     t,
    # #     error_estimates_res_mean[:, 0] + error_estimates_res_std[:, 0],
    # #     color="black",
    # #     linestyle="dashed",
    # # )
    ax[i].semilogy(
        t,
        true_error ** 2,
        color=reference_color,
        linestyle=reference_linestyle,
    )
    ax[i].set_ylim(LOG_YLIM)
    ax[i].set_ylabel("$\\log \\mathbb{E}[\\delta(Y)](t)$", fontsize="x-small")
    ax[i].set_title(
        f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    )
    ax[i].axhline(TOL, color=reference_color)
    i = next(idx)

    # Just the STD.
    ax[i].semilogy(
        t,
        error_estimates_res_mean[:, 0] ** 2 + error_estimates_res_std[:, 0] ** 2,
        color="black",
    )
    # # ax[i].semilogy(
    # #     t,
    # #     error_estimates_res_mean[:, 0] + error_estimates_res_std[:, 0],
    # #     color="black",
    # #     linestyle="dashed",
    # # )
    ax[i].semilogy(
        t,
        error_estimates_res_std[:, 0] ** 2,
        color="black",
    )
    ax[i].semilogy(
        t,
        true_error ** 2,
        color=reference_color,
        linestyle=reference_linestyle,
    )
    ax[i].set_ylim(LOG_YLIM)
    ax[i].set_ylabel("$\\log \\mathbb{V}[\\delta(Y)](t)$", fontsize="x-small")
    ax[i].set_title(
        f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    )
    ax[i].axhline(TOL, color=reference_color)
    i = next(idx)

    # Just the STD.
    ax[i].semilogy(
        t,
        error_estimates_res_std[:, 0] ** 2,
        color="black",
    )
    # # ax[i].semilogy(
    # #     t,
    # #     error_estimates_res_mean[:, 0] + error_estimates_res_std[:, 0],
    # #     color="black",
    # #     linestyle="dashed",
    # # )
    ax[i].semilogy(
        t,
        error_estimates_res_std[:, 0] ** 2,
        color="black",
    )
    ax[i].semilogy(
        t,
        true_error ** 2,
        color=reference_color,
        linestyle=reference_linestyle,
    )
    ax[i].set_ylim(LOG_YLIM)
    ax[i].set_ylabel("$\\log \\|\\delta(Y)(t)\\|$", fontsize="x-small")
    ax[i].set_title(
        f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    )
    ax[i].axhline(TOL, color=reference_color)
    i = next(idx)

    # Sixth row: steps from refinement with residual only
    # for t in grid_refine_via_res:
    #     ax[i].axvline(t, color="black", linewidth=0.25)
    # ax[i].set_title(
    #     f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    # )
    # i = next(idx)
    #
    # # # Seventh row: steps from refinement with residual only
    # # for t in grid_refine_via_prob_res:
    # #     ax[i].axvline(t, color="black", linewidth=0.5)
    # # ax[i].set_title(
    # #     f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    # # )
    # # i = next(idx)

    # Clean up: remove all ticks for now and show the plot
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlim((bvp.t0, bvp.tmax))

    plt.show()

