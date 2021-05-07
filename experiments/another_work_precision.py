"""Try out probsolve_bvp."""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars, filtsmooth, diffeq
from probnum._randomvariablelist import _RandomVariableList
from bvps import problem_examples, bridges, solver
from tqdm import tqdm
import pandas as pd

from probnumeval import timeseries
from probnum import random_variables as randvars


from scipy.integrate import solve_bvp

# Easy aliases
anees = timeseries.average_normalized_estimation_error_squared
rmse = timeseries.root_mean_square_error


TMAX = 1.0
XI = 0.01


bvp = problem_examples.problem_7_second_order(xi=XI)
bvp1st = problem_examples.problem_7(xi=XI)


print(bvp1st.y0, bvp1st.ymax)
print(bvp1st.L, bvp1st.R)
TOL = 1e-5


initial_grid = np.linspace(bvp.t0, bvp.tmax, 10)
initial_guess = np.zeros((2, len(initial_grid)))
refsol = solve_bvp(bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=TOL)
refsol_fine = solve_bvp(
    bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=1e-12
)
bvp.solution = refsol_fine.sol

q = 5


results = {}

testlocations = np.linspace(bvp.t0, bvp.tmax, 500)


for q in [3, 4, 5, 6]:
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

    integ = bridges.GaussMarkovBridge(ibm, bvp)

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

    for TOL in 10.0 ** -(np.arange(1.0, 6.0)):
        print("tol", TOL)
        posterior_generator = solver.probsolve_bvp(
            bvp=bvp,
            bridge_prior=integ,
            initial_grid=initial_grid,
            atol=1 * TOL,
            rtol=1 * TOL,
            insert="double",
            which_method="ekf",
            maxit=5,
            ignore_bridge=False,
            which_errors="probabilistic_defect",
            refinement="tolerance",
            initial_sigma_squared=1e1,
        )

        # for idx, x in enumerate(posterior_generator):
        #     print(x)
        #     if idx > 1:
        #         assert False
        for idx, (
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
            print(len(post.locations), end=" - ")
        print()

        solution = post

        testlocations = np.linspace(bvp.t0, bvp.tmax)
        reference_solution = lambda *args, **kwargs: refsol_fine.sol(*args, **kwargs)[
            0
        ].T.reshape((-1, 1))

        solution_mean = lambda *args, **kwargs: solution(*args, **kwargs).mean

        chi2 = anees(solution, reference_solution, testlocations) / ssq

        error = rmse(solution_mean, reference_solution, testlocations)
        results[q][TOL] = {}
        results[q][TOL]["chi2"] = chi2
        results[q][TOL]["error"] = error
        results[q][TOL]["N"] = len(solution.locations)

print(results)

import json

with open("./data/problem7_problem_work_precision.json", "w") as outfile:
    json.dump(results, outfile)
