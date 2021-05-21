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
from probnumeval import timeseries
from tqdm import tqdm

from bvps import bvp_solver, ode_measmods, problem_examples

bvp = problem_examples.problem_24_second_order(xi=2.5e-2)
ibm = statespace.IBM(
    ordint=3,
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

# Plotting parameters
t = np.linspace(bvp.t0, bvp.tmax, 200)
titles = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOG_YLIM = (1e-12, 1e5)
plt.style.use(
    [
        "./visualization/stylesheets/science.mplstyle",
        "./visualization/stylesheets/misc/grid.mplstyle",
        "./visualization/stylesheets/color/high-contrast.mplstyle",
        "./visualization/stylesheets/8pt.mplstyle",
    ]
)
mpl.rcParams["lines.linewidth"] = 0.5


# Reference solution
TOL = 1e-5
initial_grid = np.linspace(bvp.t0, bvp.tmax, 150)
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


N = 32

for TOL in [1e-1, 1e-2, 1e-3, 1e-4]:
    initial_grid = np.linspace(bvp.t0, bvp.tmax, N)
    initial_guess = np.ones((len(initial_grid), bvp.dimension))
    solver = bvp_solver.BVPSolver.from_default_values_std_refinement(
        ibm,
        use_bridge=True,
        initial_sigma_squared=1e2,
        normalise_with_interval_size=False,
    )
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
    for i, (kalman_posterior, sigma_squared) in enumerate(solution_gen):
        print(i, ":", len(kalman_posterior.locations))
        if i > 5:
            print("fail")
            break

    refsol = lambda x: reference_posterior(x).mean[:, 0]
    approxsol = lambda x: kalman_posterior(x).mean[:, 0]
    rmse = timeseries.root_mean_square_error(approxsol, refsol, t)

    T = kalman_posterior.locations
    print(
        "STD",
        TOL,
        rmse,
        len(T),
        np.amax(np.diff(T)),
        np.amin(np.diff(T)),
        np.amax(np.diff(T)) / np.amin(np.diff(T)),
    )
    #
    # solver = bvp_solver.BVPSolver.from_default_values(
    #     ibm, use_bridge=True, initial_sigma_squared=1e2, normalise_with_interval_size=False
    # )
    # solution_gen = solver.solution_generator(
    #     bvp,
    #     atol=TOL,
    #     rtol=TOL,
    #     initial_grid=initial_grid,
    #     initial_guess=None,
    #     maxit_ieks=MAXIT,
    #     maxit_em=1,
    #     yield_ieks_iterations=False,
    # )
    # # Skip initialisation
    # for i, (kalman_posterior, sigma_squared) in enumerate(solution_gen):
    #     print(i, ":", len(kalman_posterior.locations))
    #     if i > 5:
    #         print("fail")
    #         break
    #
    # refsol = lambda x: reference_posterior(x).mean[:, 0]
    # approxsol = lambda x: kalman_posterior(x).mean[:, 0]
    # rmse = timeseries.root_mean_square_error(approxsol, refsol, t)
    # print("Res", TOL, rmse, len(kalman_posterior.locations))
    #
    #
    #
    #
    #
    # solver = bvp_solver.BVPSolver.from_default_values_probabilistic_refinement(
    #     ibm, use_bridge=True, initial_sigma_squared=1e2, normalise_with_interval_size=False
    # )
    # solution_gen = solver.solution_generator(
    #     bvp,
    #     atol=TOL,
    #     rtol=TOL,
    #     initial_grid=initial_grid,
    #     initial_guess=None,
    #     maxit_ieks=MAXIT,
    #     maxit_em=1,
    #     yield_ieks_iterations=False,
    # )
    # # Skip initialisation
    # for i, (kalman_posterior, sigma_squared) in enumerate(solution_gen):
    #     print(i, ":", len(kalman_posterior.locations))
    #     if i > 5:
    #         print("fail")
    #         break
    #
    # refsol = lambda x: reference_posterior(x).mean[:, 0]
    # approxsol = lambda x: kalman_posterior(x).mean[:, 0]
    # rmse = timeseries.root_mean_square_error(approxsol, refsol, t)
    # print("PRes", TOL, rmse, len(kalman_posterior.locations))
