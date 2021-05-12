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
bvp = problem_examples.problem_7_second_order(xi=1e-4)
ibm = statespace.IBM(
    ordint=4,
    spatialdim=bvp.dimension,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
measmod = ode_measmods.from_second_order_ode(bvp, ibm)


# Solver and solver parameters
solver = bvp_solver.BVPSolver.from_default_values(
    ibm, use_bridge=False, initial_sigma_squared=1e2, normalise_with_interval_size=False
)
MAXIT = 10
TOL = 1e-6

# Plotting parameters
t = np.linspace(bvp.t0, bvp.tmax, 200)
titles = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOG_YLIM = (1e-16, 1e6)
plt.style.use(
    [
        "./visualization/stylesheets/science.mplstyle",
        # "./visualization/stylesheets/misc/grid.mplstyle",
        "./visualization/stylesheets/color/high-contrast.mplstyle",
        "./visualization/stylesheets/8pt.mplstyle",
        "./visualization/stylesheets/23_tile_jmlr.mplstyle",
    ]
)
# mpl.rcParams['lines.linewidth'] = 0.5

COLOR = "darkgreen"
SECOND_COLOR = "blue"


# Reference solution

initial_grid = np.linspace(bvp.t0, bvp.tmax, 256)
initial_guess = np.ones((len(initial_grid), bvp.dimension))
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
reference_posterior, _ = next(solution_gen)

# Set up all 7 subplots
fig, axes = plt.subplots(
    nrows=4,
    ncols=3,
    sharex="col",
    sharey="row",
    dpi=500,
    constrained_layout=True,
    gridspec_kw={"height_ratios": [8, 8, 8, 8]},
)

for axis_index, (N, ax) in enumerate(zip([5, 5 ** 2, 5 ** 3], axes.T)):
    # ax[0].set_title(f"$N={N}$")
    initial_grid = np.linspace(bvp.t0, bvp.tmax, N)
    initial_guess = np.ones((len(initial_grid), bvp.dimension))

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

    idx = itertools.count()
    i = next(idx)

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
    ax[i].semilogy(t, error_estimates_std ** 2, "-", color="black")
    ax[i].fill_between(
        t, 1e-16 * np.ones_like(t), error_estimates_std ** 2, color=COLOR, alpha=0.2
    )
    if axis_index == 0:
        ax[i].set_ylabel("Variance")
    ax[i].semilogy(
        t,
        true_error ** 2,
        color="black",
        linestyle=reference_linestyle,
    )
    ax[i].set_ylim(LOG_YLIM)
    ax[i].set_yticks([LOG_YLIM[0], 1e-5, LOG_YLIM[1]])
    ax[i].set_title(
        f"\\bf {titles[i]}{str(N)}", loc="left", fontweight="bold", fontsize="small"
    )

    i = next(idx)

    # Next row: Solution and uncertainties
    ax[i].plot(
        t,
        reference_evaluated,
        color="black",
        linestyle=reference_linestyle,
    )
    ax[i].plot(t, posterior_mean[:, 0], color="black")
    ax[i].fill_between(
        t,
        posterior_mean[:, 0] - 3 * posterior_std[:, 0],
        posterior_mean[:, 0] + 3 * posterior_std[:, 0],
        alpha=0.2,
        color=COLOR,
        linewidth=0,
    )
    ax[i].plot(t, posterior_mean[:, 1], color="black", linestyle="dashed")
    ax[i].fill_between(
        t,
        posterior_mean[:, 1] - 3 * posterior_std[:, 1],
        posterior_mean[:, 1] + 3 * posterior_std[:, 1],
        alpha=0.2,
        color=COLOR,
        linewidth=0,
    )
    if axis_index == 0:
        ax[i].set_ylabel("Solution")
    ax[i].set_ylim((-3, 5))
    ax[i].set_yticks([-3, 1, 5])
    ax[i].set_title(
        f"\\bf {titles[i]}{str(N)}", loc="left", fontweight="bold", fontsize="small"
    )
    i = next(idx)

    # Next row: Residual and uncertainties
    ax[i].plot(
        t,
        residual_mean[:, 0],
        color="black",
    )
    ax[i].plot(
        t,
        np.zeros_like(t),
        color="black",
        linestyle=reference_linestyle,
    )
    ax[i].fill_between(
        t,
        residual_mean[:, 0] - 3 * residual_std[:, 0],
        residual_mean[:, 0] + 3 * residual_std[:, 0],
        alpha=0.2,
        color=SECOND_COLOR,
        linewidth=0,
    )
    if axis_index == 0:
        ax[i].set_ylabel("Residual")
    ax[i].set_ylim((-2000, 2000))
    ax[i].set_yticks([-2000, 0, 2000])

    ax[i].set_title(
        f"\\bf {titles[i]}{str(N)}", loc="left", fontweight="bold", fontsize="small"
    )

    i = next(idx)

    # Next row: residual and uncertainties
    assert t.shape == error_estimates_res_mean[:, 0].shape
    ax[i].semilogy(
        t,
        error_estimates_res_mean[:, 0] ** 2,
        color="black",
    )

    ax[i].semilogy(
        t,
        error_estimates_res_mean[:, 0] ** 2 + error_estimates_res_std[:, 0] ** 2,
        color="black",
        linestyle="dashed",
    )

    ax[i].fill_between(
        t,
        1e-16 * np.ones_like(t),
        error_estimates_res_mean[:, 0] ** 2,
        color="black",
        alpha=0.2,
    )
    ax[i].fill_between(
        t,
        error_estimates_res_mean[:, 0] ** 2,
        error_estimates_res_mean[:, 0] ** 2 + error_estimates_res_std[:, 0] ** 2,
        color=SECOND_COLOR,
        alpha=0.2,
    )

    ax[i].semilogy(
        t,
        true_error ** 2,
        color="black",
        linestyle=reference_linestyle,
    )
    ax[i].set_ylim(LOG_YLIM)
    ax[i].set_yticks([LOG_YLIM[0], 1e-5, LOG_YLIM[1]])
    if axis_index == 0:
        ax[i].set_ylabel("Residual")
    ax[i].set_title(
        f"\\bf {titles[i]}{str(N)}", loc="left", fontweight="bold", fontsize="small"
    )
    i = next(idx)

    # Clean up: remove all ticks for now and show the plot
    for a in ax:
        a.set_xticks([bvp.t0, bvp.t0 + 0.5 * (bvp.tmax - bvp.t0), bvp.tmax])
        # a.set_yticks([])
        a.set_xlim((bvp.t0, bvp.tmax))
for ax in axes[-1]:
    ax.set_xlabel("Time $t$")
fig.align_ylabels()
plt.savefig(f"errorestimates{N}.pdf")
plt.show()
