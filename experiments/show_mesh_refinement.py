"""Try out probsolve_bvp."""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars, filtsmooth, diffeq
from probnum._randomvariablelist import _RandomVariableList
from bvps import problem_examples, bridges, solver
from tqdm import tqdm
import pandas as pd

import itertools
from probnum import random_variables as randvars


from scipy.integrate import solve_bvp

TOL = 1e-2


TMAX = 1.0
XI = 1e-7
bvp = problem_examples.problem_7_second_order(xi=XI)
bvp1st = problem_examples.problem_7(xi=XI)

print(bvp1st.y0, bvp1st.ymax)
print(bvp1st.L, bvp1st.R)


initial_grid = np.linspace(bvp.t0, bvp.tmax, 10)
initial_guess = np.zeros((2, len(initial_grid)))
refsol = solve_bvp(bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=TOL)
refsol_fine = solve_bvp(
    bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=1e-12
)
bvp.solution = refsol_fine.sol

q = 6


ibm = statespace.IBM(
    ordint=q,
    spatialdim=1,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)

integ = bridges.GaussMarkovBridge(ibm, bvp)


posterior_generator = solver.probsolve_bvp(
    bvp=bvp,
    bridge_prior=integ,
    initial_grid=initial_grid,
    atol=1 * TOL,
    rtol=1 * TOL,
    insert="double",
    which_method="ekf",
    maxit=3,
    ignore_bridge=False,
    which_errors="probabilistic_defect",
    refinement="tolerance",
    initial_sigma_squared=1e1,
)

# Thin lines everywhere -- lots of plots to show!
titles = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
evalgrid = np.linspace(bvp.t0, bvp.tmax, 250, endpoint=True)
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
# plt.rcParams.update({"lines.linewidth": 0.75})


for iteration, (
    post,
    ssq,
    integral_error,
    kalpost,
    candidates,
    h,
    quotient,
    sigmas,
    insert_one,
    insert_two,
    measmod,
) in enumerate(posterior_generator):

    print(len(post.locations))
    print(len(refsol_fine.x))
    print(len(refsol.x))
    print()
    # Set up all 7 subplots
    fig, ax = plt.subplots(
        nrows=5,
        sharex=True,
        dpi=200,
        figsize=(2, 3),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [8, 8, 8, 8, 1]},
    )
    idx = itertools.count()
    i = next(idx)

    # Evaluate the posterior at the grid
    posterior_evaluations = kalpost(evalgrid)
    posterior_mean = posterior_evaluations.mean
    posterior_std = posterior_evaluations.std

    # Evaluate the residual
    residual_evaluations = _RandomVariableList(
        [measmod.forward_rv(rv, t)[0] for rv, t in zip(posterior_evaluations, evalgrid)]
    )
    residual_mean = residual_evaluations.mean
    residual_std = residual_evaluations.std

    # Evaluate the reference solution and errors
    reference_evaluated = refsol_fine.sol(evalgrid).T
    reference_color = "crimson"
    reference_linestyle = "dotted"
    true_error_dy = np.abs(reference_evaluated[:, 1] - posterior_mean[:, 1])

    # Extract plotting residuals and uncertainties
    error_estimates_std = np.abs(posterior_std[:, 1])
    error_estimates_res_mean = np.abs(residual_mean)
    error_estimates_res_std = np.abs(residual_std)

    # Select grids to be plotted
    grid_refine_via_std = post.locations  # dummy
    grid_refine_via_prob_res = post.locations  # dummy
    grid_refine_via_res = post.locations

    # # First row: std-refinement grid
    # for t in grid_refine_via_std:
    #     ax[i].axvline(t, color="black", linewidth=0.5)
    # ax[i].set_title(
    #     f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    # )
    # i = next(idx)

    # Second row: uncertainties derived from std
    # ax[i].semilogy(evalgrid, error_estimates_std, ":", color="black")
    ax[i].semilogy(evalgrid, ssq * error_estimates_std ** 2, "-", color="black")
    ax[i].set_ylabel("$\\log \\mathbb{V}[Y_1](t)$", fontsize="x-small")
    ax[i].semilogy(
        evalgrid,
        true_error_dy ** 2,
        color=reference_color,
        linestyle=reference_linestyle,
    )
    ax[i].axhline(TOL, color=reference_color)
    ax[i].set_title(
        f"$\\bf {titles[i]}{iteration}$",
        loc="left",
        fontweight="bold",
        fontsize="x-small",
    )
    ax[i].set_ylim(LOG_YLIM)
    i = next(idx)

    # Third row: Solution and uncertainties
    ax[i].plot(
        evalgrid,
        reference_evaluated[:, 1],
        color=reference_color,
        linestyle=reference_linestyle,
    )
    ax[i].plot(evalgrid, posterior_mean[:, 1], color="black")
    ax[i].fill_between(
        evalgrid,
        posterior_mean[:, 1] - 3 * np.sqrt(ssq) * posterior_std[:, 1],
        posterior_mean[:, 1] + 3 * np.sqrt(ssq) * posterior_std[:, 1],
        alpha=0.2,
        color="black",
        linewidth=0,
    )
    ax[i].set_ylabel("$Y_1(t)$", fontsize="x-small")
    ax[i].set_title(
        f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    )
    i = next(idx)

    # Fourth row: Residual and uncertainties
    ax[i].plot(
        evalgrid,
        residual_mean[:, 0],
        color="black",
    )
    ax[i].plot(
        evalgrid,
        np.zeros_like(evalgrid),
        color=reference_color,
        linestyle=reference_linestyle,
    )
    ax[i].fill_between(
        evalgrid,
        residual_mean[:, 0] - 3 * np.sqrt(ssq) * residual_std[:, 0],
        residual_mean[:, 0] + 3 * np.sqrt(ssq) * residual_std[:, 0],
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
        evalgrid,
        error_estimates_res_mean[:, 0] ** 2,
        color="black",
    )
    # ax[i].semilogy(
    #     evalgrid,
    #     error_estimates_res_mean[:, 0] + error_estimates_res_std[:, 0],
    #     color="black",
    #     linestyle="dashed",
    # )
    ax[i].semilogy(
        evalgrid,
        error_estimates_res_mean[:, 0] ** 2 + ssq * error_estimates_res_std[:, 0] ** 2,
        color="black",
        linestyle="dotted",
    )
    ax[i].semilogy(
        evalgrid,
        true_error_dy ** 2,
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

    # Sixth row: steps from refinement with residual only
    for t in grid_refine_via_res:
        ax[i].axvline(t, color="black", linewidth=0.25)
    ax[i].set_title(
        f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    )
    i = next(idx)

    # # Seventh row: steps from refinement with residual only
    # for t in grid_refine_via_prob_res:
    #     ax[i].axvline(t, color="black", linewidth=0.5)
    # ax[i].set_title(
    #     f"$\\bf {titles[i]}$", loc="left", fontweight="bold", fontsize="x-small"
    # )
    # i = next(idx)

    # Clean up: remove all ticks for now and show the plot
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlim((bvp.t0, bvp.tmax))

    plt.show()
