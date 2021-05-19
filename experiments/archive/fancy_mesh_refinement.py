"""Try out probsolve_bvp."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from probnum import diffeq, filtsmooth
from probnum import random_variables as randvars
from probnum import randvars, statespace
from probnum._randomvariablelist import _RandomVariableList
from scipy.integrate import solve_bvp
from tqdm import tqdm

from bvps import (
    BoundaryValueProblem,
    MyIteratedDiscreteComponent,
    MyKalman,
    MyStoppingCriterion,
    WrappedIntegrator,
    bratus,
    bratus_second_order,
    from_ode,
    generate_samples,
    matlab_example,
    matlab_example_second_order,
    new_grid,
    new_grid2,
    problem_7,
    problem_7_second_order,
    probsolve_bvp,
    r_example,
    split_grid,
)

TOL = 1e-3

# bvp = r_example(xi=0.01)
# # bvp = matlab_example()

# bvp = bratus_second_order()
# bvp1st = bratus()


# bvp = matlab_example_second_order(tmax=1)
# bvp1st = matlab_example(tmax=1)


# bvp = problem_7_second_order(xi=0.1)
# bvp1st = problem_7(xi=0.1)

initial_grid = np.union1d(
    np.linspace(bvp.t0, bvp.tmax, 10), np.linspace(bvp.t0, bvp.t0 + 0.2, 10)
)
initial_guess = np.zeros((2, len(initial_grid)))
refsol = solve_bvp(bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=TOL)


# initial_grid = refsol.x

# plt.plot(refsol.x, np.ones_like(refsol.x), ".")
# plt.show()
# print(refsol.x)
# assert False


q = 5

ibm = statespace.IBM(
    ordint=q,
    spatialdim=1,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)

integ = WrappedIntegrator(ibm, bvp)

posterior = probsolve_bvp(
    bvp=bvp,
    bridge_prior=integ,
    initial_grid=initial_grid,
    atol=1 * TOL,
    rtol=1 * TOL,
    insert="double",
    which_method="iekf",
    maxit=5,
    ignore_bridge=False,
    which_errors="defect",
    refinement="tolerance",
)


print(
    "Current information: EM in the filter sucks, EM on iterated filtsmooth level is okay, no EM is fine too."
)


evalgrid = np.linspace(bvp.t0, bvp.tmax, 150, endpoint=True)

for idx, (post, ssq, errors, kalpost, candidates, h) in enumerate(posterior):
    print(
        "Why is the filtering posterior soooo bad even if the smoothing posterior is alright?"
    )
    post2 = diffeq.KalmanODESolution(kalpost.filtering_posterior)

    # print(post.states[0].mean)
    # print(kalpost.filtering_posterior.states[0].mean)
    # post = post.filtering_solution
    # print(kalpost.states[0].mean)
    # print(kalpost.states[0].std)
    # print()
    plt.style.use(
        [
            "./visualization/science.mplstyle",
            # "./visualization/notebook.mplstyle"
        ]
    )

    fig, ax = plt.subplots(nrows=3, sharex=True, dpi=200)
    evaluated = post(evalgrid)
    m = evaluated.mean[:, 0]
    s = evaluated.std[:, 0]  # * np.sqrt(ssq)

    evaluated2 = post2(evalgrid)
    m2 = evaluated2.mean[:, 0]
    s2 = evaluated2.std[:, 0]  # * np.sqrt(ssq)

    discrepancy = np.abs(refsol.sol(evalgrid).T[:, 0] - m)
    ax[0].plot(evalgrid, m, color="k")
    ax[0].plot(evalgrid, m2, color="orange")
    ax[0].plot(
        evalgrid, refsol.sol(evalgrid).T[:, 0], color="steelblue", linestyle="dashed"
    )
    ax[0].plot(
        evalgrid, bvp.solution(evalgrid).T[:, 0], color="red", linestyle="dotted"
    )

    discrepancy = np.abs(bvp.solution(evalgrid)[0] - m)
    scipy_discrepancy = np.abs(refsol.sol(evalgrid)[0] - bvp.solution(evalgrid)[0])

    # ax[1].semilogy(evalgrid, s, color="k", label="Uncertainty")
    ax[1].semilogy(
        candidates,
        np.linalg.norm(errors, axis=1),
        ".",
        color="darksalmon",
        label="Estimated error",
    )
    # ax[1].semilogy(
    #     candidates[np.linalg.norm(errors, axis=1) > np.median(np.linalg.norm(errors, axis=1))],
    #     np.linalg.norm(errors, axis=1)[np.linalg.norm(errors, axis=1) > np.median(np.linalg.norm(errors, axis=1))],
    #     "+",
    #     color="green",
    #     label="Error",
    # )

    ax[1].semilogy(
        evalgrid, discrepancy, linestyle="dashed", color="steelblue", label="True Error"
    )
    ax[1].semilogy(
        evalgrid,
        scipy_discrepancy,
        color="gray",
        linestyle="dotted",
        label="Scipy error",
    )
    ax[1].axhline(TOL, color="black")

    ax[2].semilogy(post.locations[:-1], np.diff(post.locations), "x", color="k")
    ax[2].semilogy(refsol.x[:-1], np.diff(refsol.x), ".", color="steelblue")
    # ax[0].set_ylim((-1.5, 3.5))
    ax[1].set_ylim((1e-14, 1e1))
    ax[2].set_ylim((1e-4, 1e0))
    ax[1].legend(frameon=False)

    ax[2].set_xlabel("Time")
    ax[0].set_ylabel("Solution")
    ax[1].set_ylabel("Error / Estimation")
    ax[2].set_ylabel("Stepsize")
    ax[0].set_title(
        f"Refinement {idx + 1}: $N={len(post.locations)}$ Points | Scipy {len(refsol.x)}"
    )
    fig.align_ylabels()
    plt.show()
