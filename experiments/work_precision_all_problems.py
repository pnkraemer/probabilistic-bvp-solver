"""This is the master result."""

import time

import numpy as np
import tqdm
from probnum import diffeq, statespace
from probnumeval import timeseries
from scipy.integrate import solve_bvp

from bvps import bvp_solver, problem_examples

# Load problems
XI = 0.02
bvp = problem_examples.problem_7_second_order(xi=XI)
bvp1st = problem_examples.problem_7(xi=XI)

# Compute reference solution
initial_grid = np.linspace(bvp.t0, bvp.tmax, 10)
initial_guess = np.zeros((bvp1st.dimension, len(initial_grid)))
refsol_fine = solve_bvp(
    bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=1e-7
)
assert refsol_fine.success
bvp.solution = refsol_fine.sol
reference_solution = lambda *args: bvp.solution(*args)[0][:, None]

# Set up empty dictionary and determine the locations for computation of anees and rmse
results = {}
testlocations = np.linspace(bvp.t0, bvp.tmax, 400)


ORDERS = list(reversed([3, 4, 5]))
TOLERANCES = list(reversed([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]))

for q in tqdm.tqdm(ORDERS):

    # Set up dictionary for the results for given q
    results[q] = {}

    for tol in tqdm.tqdm(TOLERANCES):

        # Set up dictionary for the results for given tolerance
        results[q][tol] = {}

        # Solve the BVP
        ibm = statespace.IBM(
            ordint=q,
            spatialdim=bvp.dimension,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
        solver = bvp_solver.BVPSolver.from_default_values_std_refinement(
            ibm, initial_sigma_squared=1e11
        )

        start_time = time.time()
        # We dont need initial guess nor bridge because the problem is linear.
        initial_posterior, _ = solver.compute_initialisation(
            bvp, initial_grid, initial_guess=None, use_bridge=False
        )

        solution_gen = solver.solution_generator(
            bvp,
            atol=tol,
            rtol=tol,
            initial_posterior=initial_posterior,
            maxit_ieks=1,
            maxit_em=1,
            yield_ieks_iterations=False,
        )

        for kalman_posterior, sigma_squared in solution_gen:
            pass
        runtime = time.time() - start_time

        # Transform solution into something more convenient
        solution = diffeq.KalmanODESolution(kalman_posterior)
        solution_mean = lambda *args: solution(*args).mean

        # Compute error and calibration
        chi2 = timeseries.anees(solution, reference_solution, testlocations)

        error = timeseries.root_mean_square_error(
            solution_mean, reference_solution, testlocations
        )
        N = len(solution.locations)

        # How fast would scipy be?
        start_time_scipy = time.time()
        scipy_solution = solve_bvp(
            bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=tol
        )
        runtime_scipy = time.time() - start_time_scipy
        assert scipy_solution.success

        # How accurate would scipy be?
        scipy_sol_for_rmse = lambda *args: scipy_solution.sol(*args)[0][:, None]
        error_scipy = timeseries.root_mean_square_error(
            scipy_sol_for_rmse, reference_solution, testlocations
        )

        # Save results
        results[q][tol]["chi2"] = chi2 / sigma_squared
        results[q][tol]["time"] = runtime
        results[q][tol]["error"] = error
        results[q][tol]["N"] = N
        results[q][tol]["largest_step"] = np.amax(np.diff(solution.locations))
        results[q][tol]["time_scipy"] = runtime_scipy
        results[q][tol]["error_scipy"] = error_scipy
        results[q][tol]["N_scipy"] = len(scipy_solution.x)
        results[q][tol]["largest_step_scipy"] = np.amax(np.diff(scipy_solution.x))

print(results)

import json

with open("./data/problem7_problem_work_precision.json", "w") as outfile:
    json.dump(results, outfile)
