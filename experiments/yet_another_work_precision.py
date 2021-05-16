"""Try out probsolve_bvp."""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars, filtsmooth, diffeq
from probnum._randomvariablelist import _RandomVariableList
from bvps import problem_examples, bridges, bvp_solver
from tqdm import tqdm
import pandas as pd

from probnumeval import timeseries
from probnum import random_variables as randvars


from scipy.integrate import solve_bvp
import time

# Easy aliases
anees = timeseries.average_normalized_estimation_error_squared
rmse = timeseries.root_mean_square_error


TMAX = 1.0
XI = 0.001


bvp = problem_examples.bratus_second_order()
bvp1st = problem_examples.bratus()

TOL = 1e-5

initial_grid = np.linspace(bvp.t0, bvp.tmax, 100)
initial_guess = np.zeros((len(initial_grid), 2)).T
refsol = solve_bvp(bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=TOL)
refsol_fine = solve_bvp(
    bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=1e-10
)
print(refsol_fine.success, len(refsol_fine.x))
bvp.solution = refsol_fine.sol


results = {}

testlocations = np.linspace(bvp.t0, bvp.tmax, 500)


for q, initial_N, maxit_IEKS in zip([6, 5, 4], [10, 10, 10], [2, 5, 5]):
    print()
    print()
    print("q", q)
    print()

    results[q] = {}

    ibm = statespace.IBM(
        ordint=q,
        spatialdim=1,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
    # ibm.equivalent_discretisation_preconditioned._proc_noise_cov_cholesky *= 1e5

    # initial_grid = np.linspace(bvp.t0, bvp.tmax, 2)

    # print(len(refsol.x))
    # reference_solution = lambda *args, **kwargs: refsol_fine.sol(*args, **kwargs)[
    #     0
    # ].T.reshape((-1, 1))
    # scipy_sol = lambda *args, **kwargs: refsol.sol(*args, **kwargs)[0].T.reshape(
    #     (-1, 1)
    # )

    # error = rmse(scipy_sol, reference_solution, testlocations)
    # print("Scipyerror:", error)

    evalgrid = np.linspace(bvp.t0, bvp.tmax, 250, endpoint=True)

    for tol_order in np.arange(1.0, 9.0):
        TOL = 10.0 ** (-tol_order)

        print("tol", TOL)
        solver = bvp_solver.BVPSolver.from_default_values(
            ibm, initial_sigma_squared=1e2, normalise_with_interval_size=False
        )

        initial_grid = np.linspace(bvp.t0, bvp.tmax, initial_N)

        initial_guess = np.ones((len(initial_grid), bvp.dimension))

        initial_posterior, sigma_squared = solver.compute_initialisation(
            bvp, initial_grid, initial_guess=initial_guess, use_bridge=True
        )

        solution_gen = solver.solution_generator(
            bvp,
            atol=TOL,
            rtol=TOL,
            initial_posterior=initial_posterior,
            maxit_ieks=maxit_IEKS,
            maxit_em=1,
            yield_ieks_iterations=False,
        )

        start_time = time.time()
        for post, ssq in solution_gen:
            print(len(post.locations))
        end_time = time.time() - start_time
        solution = diffeq.KalmanODESolution(post)

        testlocations = np.linspace(bvp.t0, bvp.tmax)
        reference_solution = lambda *args, **kwargs: bvp.solution(*args, **kwargs)[0].reshape((-1, 1))
        # plt.plot(testlocations, reference_solution(testlocations))
        # plt.plot(testlocations, solution(testlocations).mean[:, 0])
        # plt.show()
        solution_mean = lambda *args, **kwargs: solution(*args, **kwargs).mean[:, :1].reshape((-1, 1))

        print(ssq)
        chi2 = anees(solution, reference_solution, testlocations, damping=1e-30) / ssq

        error = rmse(solution_mean, reference_solution, testlocations)
        results[q][TOL] = {}
        results[q][TOL]["chi2"] = chi2
        results[q][TOL]["error"] = error
        results[q][TOL]["N"] = len(solution.locations)
        results[q][TOL]["time"] = end_time
        print(chi2, error, end_time)
print(results)

import json

with open("./data/yet_another_work_precision.json", "w") as outfile:
    json.dump(results, outfile)
