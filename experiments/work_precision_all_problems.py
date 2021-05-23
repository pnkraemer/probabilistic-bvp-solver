"""This is the master result."""

import time

import numpy as np
import tqdm
from probnum import diffeq, statespace
from probnumeval import timeseries
from scipy.integrate import solve_bvp

from bvps import bvp_solver, problem_examples

# Load problems

# Set up empty dictionary and determine the locations for computation of anees and rmse
results = {}

problems = [
    problem_examples.problem_23_second_order(xi=0.25),
    problem_examples.problem_32_fourth_order(xi=0.25),
    problem_examples.problem_24_second_order(xi=0.5, gamma=1.4),
    problem_examples.problem_28_second_order(xi=0.4),
    problem_examples.problem_7_second_order(xi=0.1),
]
labels = [
    "23rd Problem",
    "32nd Problem",
    "24th Problem",
    "28th Problem",
    "7th Problem",
]

for bvp, label in zip(problems, labels):
    assert bvp.dimension == 1
    results[label] = {}
    ORDERS = list(reversed([6]))
    TOLERANCES = list(reversed([1e-1, 1e-6]))

    bvp1st = bvp.to_first_order()

    # Compute reference solution
    initial_grid = np.linspace(bvp.t0, bvp.tmax, 50)
    initial_guess = np.ones((bvp1st.dimension, len(initial_grid)))
    refsol_fine = solve_bvp(
        bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=1e-7
    )
    assert refsol_fine.success

    initial_grid = np.linspace(bvp.t0, bvp.tmax, 5)
    initial_guess = np.ones((bvp1st.dimension, len(initial_grid)))

    bvp.solution = refsol_fine.sol
    reference_solution = lambda *args: bvp.solution(*args)[0][:, None]
    testlocations = np.linspace(bvp.t0, bvp.tmax, 400)

    for q in ORDERS:

        # Set up dictionary for the results for given q
        results[label][q] = {}

        for tol in TOLERANCES:

            # Set up dictionary for the results for given tolerance
            results[label][q][tol] = {}

            # Solve the BVP
            ibm = statespace.IBM(
                ordint=q,
                spatialdim=bvp.dimension,
                forward_implementation="sqrt",
                backward_implementation="sqrt",
            )
            solver = bvp_solver.BVPSolver.from_default_values_std_refinement(
                ibm, initial_sigma_squared=1e12, normalise_with_interval_size=False
            )

            start_time = time.time()
            # We dont need initial guess nor bridge because the problem is linear.
            initial_posterior, _ = solver.compute_initialisation(
                bvp, initial_grid, initial_guess=None, use_bridge=True
            )

            solution_gen = solver.solution_generator(
                bvp,
                atol=tol,
                rtol=tol,
                initial_posterior=initial_posterior,
                maxit_ieks=3,
                maxit_em=1,
                yield_ieks_iterations=False,
            )

            for kalman_posterior, sigma_squared in solution_gen:
                print(len(kalman_posterior.locations))
            runtime = time.time() - start_time

            # Transform solution into something more convenient
            solution = diffeq.KalmanODESolution(kalman_posterior)
            solution_mean = lambda *args: solution(*args).mean

            # Compute error and calibration
            chi2 = timeseries.anees(solution, reference_solution, testlocations, damping=1e-15)
            # chi2 = timeseries.average_normalized_estimation_error_squared(
            #     solution, reference_solution, testlocations
            # )

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
            results[label][q][tol]["chi2"] = chi2 / sigma_squared
            results[label][q][tol]["time"] = runtime
            results[label][q][tol]["error"] = error
            results[label][q][tol]["N"] = N
            results[label][q][tol]["largest_step"] = np.amax(
                np.diff(solution.locations)
            )
            results[label][q][tol]["time_scipy"] = runtime_scipy
            results[label][q][tol]["error_scipy"] = error_scipy
            results[label][q][tol]["N_scipy"] = len(scipy_solution.x)
            results[label][q][tol]["largest_step_scipy"] = np.amax(
                np.diff(scipy_solution.x)
            )
        print(results[label])
print(results)

import json

with open("./data/problem7_problem_work_precision.json", "w") as outfile:
    json.dump(results, outfile)


# {
#     "P24": {
#         "4": {
#             "1e-05": {
#                 "chi2": 0.028297893753976053,
#                 "time": 3.629366397857666,
#                 "error": 2.0573119144323396e-06,
#                 "N": 68,
#                 "largest_step": 0.041666666666666664,
#                 "time_scipy": 0.02076268196105957,
#                 "error_scipy": 1.3761866684111927e-08,
#                 "N_scipy": 133,
#                 "largest_step_scipy": 0.02083333333333337,
#             },
#             "0.01": {
#                 "chi2": 962.6579017975663,
#                 "time": 0.5140864849090576,
#                 "error": 0.27110358191930095,
#                 "N": 11,
#                 "largest_step": 0.25,
#                 "time_scipy": 0.018340349197387695,
#                 "error_scipy": 6.4132719043147e-05,
#                 "N_scipy": 17,
#                 "largest_step_scipy": 0.125,
#             },
#         }
#     },
#     "P20": {
#         "4": {
#             "1e-05": {
#                 "chi2": 531997.1443590235,
#                 "time": 1.2166862487792969,
#                 "error": 0.0002357300390926717,
#                 "N": 26,
#                 "largest_step": 0.08333333333333333,
#                 "time_scipy": 0.011692047119140625,
#                 "error_scipy": 1.3713853235325707e-08,
#                 "N_scipy": 63,
#                 "largest_step_scipy": 0.041666666666666664,
#             },
#             "0.01": {
#                 "chi2": 6.521290685124109,
#                 "time": 0.10383796691894531,
#                 "error": 0.004075651823709967,
#                 "N": 5,
#                 "largest_step": 0.25,
#                 "time_scipy": 0.006120204925537109,
#                 "error_scipy": 4.184176093951428e-05,
#                 "N_scipy": 9,
#                 "largest_step_scipy": 0.125,
#             },
#         }
#     },
#     "P28": {
#         "4": {
#             "1e-05": {
#                 "chi2": 1.6682791980907021,
#                 "time": 3.1519339084625244,
#                 "error": 1.0655077767472943e-05,
#                 "N": 76,
#                 "largest_step": 0.02083333333333337,
#                 "time_scipy": 0.012783050537109375,
#                 "error_scipy": 6.742478646655015e-09,
#                 "N_scipy": 77,
#                 "largest_step_scipy": 0.04166666666666674,
#             },
#             "0.01": {
#                 "chi2": 2.492391492731803,
#                 "time": 0.10341000556945801,
#                 "error": 0.022724072248440436,
#                 "N": 5,
#                 "largest_step": 0.25,
#                 "time_scipy": 0.005528688430786133,
#                 "error_scipy": 3.720102201706612e-05,
#                 "N_scipy": 10,
#                 "largest_step_scipy": 0.25,
#             },
#         }
#     },
#     "P7": {
#         "4": {
#             "1e-05": {
#                 "chi2": 10154.377495388317,
#                 "time": 2.0635318756103516,
#                 "error": 3.8896078790963406e-05,
#                 "N": 59,
#                 "largest_step": 0.05555555555555558,
#                 "time_scipy": 0.011714696884155273,
#                 "error_scipy": 4.29489909213437e-08,
#                 "N_scipy": 140,
#                 "largest_step_scipy": 0.02777777777777779,
#             },
#             "0.01": {
#                 "chi2": 3.0692166347249326,
#                 "time": 0.22914743423461914,
#                 "error": 0.01702099389512168,
#                 "N": 7,
#                 "largest_step": 0.5,
#                 "time_scipy": 0.007096529006958008,
#                 "error_scipy": 4.693010680110855e-05,
#                 "N_scipy": 19,
#                 "largest_step_scipy": 0.125,
#             },
#         }
#     },
# }
