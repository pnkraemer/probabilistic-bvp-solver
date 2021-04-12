"""Try out probsolve_bvp."""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars, filtsmooth, diffeq
from probnum._randomvariablelist import _RandomVariableList
from bvps import *
from tqdm import tqdm
import pandas as pd


from probnum import random_variables as randvars


from scipy.integrate import solve_bvp

TOL = 1e-5

# bvp = r_example(xi=0.01)
# # bvp = matlab_example()

# bvp = bratus_second_order()
# bvp1st = bratus()

TMAX = 0.5
XI = 0.0001
bvp = problem_7_second_order(xi=XI)
bvp1st = problem_7(xi=XI)


# bvp = problem_7_second_order(xi=0.1)
# bvp1st = problem_7(xi=0.1)

# initial_grid = np.union1d(np.linspace(bvp.t0, 0.3, 200), np.linspace(bvp.t0, bvp.tmax, 20))
initial_grid = np.linspace(bvp.t0, bvp.tmax, 100)
initial_guess = np.zeros((2, len(initial_grid)))
refsol = solve_bvp(bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=TOL)
# initial_grid = refsol.x.copy()

# initial_grid = refsol.x

# plt.plot(refsol.x, np.ones_like(refsol.x), ".")
# plt.show()
# print(refsol.x)
# assert False


q = 3

ibm = statespace.IBM(
    ordint=q,
    spatialdim=2,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)

integ = WrappedIntegrator(ibm, bvp1st)


# initial_grid = np.linspace(bvp.t0, bvp.tmax, 2)


posterior = probsolve_bvp(
    bvp=bvp1st,
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
)


print(
    "Current information: EM in the filter sucks, EM on iterated filtsmooth level is okay, no EM is fine too."
)


evalgrid = np.linspace(bvp.t0, bvp.tmax, 150, endpoint=True)

for idx, (post, ssq, errors, kalpost, candidates, h, quotient, sigmas) in enumerate(
    posterior
):
    print(
        "Why is the filtering posterior soooo bad even if the smoothing posterior is alright?"
    )
    post2 = post.kalman_posterior.filtering_posterior
    # print(post.locations, post2.locations)
    # print(post.states[0].mean)
    # print(kalpost.filtering_posterior.states[0].mean)
    # post = post.filtering_solution
    # print(kalpost.states[0].mean)
    # print(kalpost.states[0].std)
    # print()
    # plt.style.use(
    #     [
    #         "./visualization/science.mplstyle",
    #         # "./visualization/notebook.mplstyle"
    #     ]
    # )

    # print(ssq)

    fig, ax = plt.subplots(nrows=3, sharex=True, dpi=200)
    evaluated = post(evalgrid)
    m = evaluated.mean[:, :2]
    s = evaluated.std[:, :2] * np.sqrt(ssq)

    evaluated2 = post2(evalgrid)
    m2 = evaluated2.mean[:, :2]
    s2 = evaluated2.std[:, :2] * np.sqrt(ssq)

    t = post.locations
    m = post.states.mean[:, :2]
    s = post.states.std[:, :2]

    t2 = post2.locations
    m2 = post2.states.mean[:, :2]
    s2 = post2.states.std[:, :2]

    discrepancy = np.abs(refsol.sol(evalgrid).T[:, 0] - post(evalgrid).mean[:, 0])
    ax[0].plot(t, m, color="k")
    # ax[0].plot(t2, m2, color="orange")
    ax[0].plot(evalgrid, refsol.sol(evalgrid).T, color="steelblue", linestyle="dashed")
    # ax[0].plot(
    #     evalgrid, bvp.solution(evalgrid).T[:, 0], color="red", linestyle="dotted"
    # )

    # print(s)
    # print(s2)

    # ax[0].fill_between(t, m - 3 * s, m + 3 * s)
    # ax[0].fill_between(t2, m2 - 3 * s2, m2 + 3 * s2)

    for t in post.locations:
        ax[0].axvline(t, linewidth=0.1, color="k")

    # discrepancy = np.abs(bvp.solution(evalgrid)[0] - post(evalgrid).mean[:, 0])
    # scipy_discrepancy = np.abs(refsol.sol(evalgrid)[0] - bvp.solution(evalgrid)[0])

    # ax[1].semilogy(evalgrid, s, color="k", label="Uncertainty")
    ax[1].semilogy(
        candidates,
        quotient,
        ".",
        color="darksalmon",
        label="Estimated error",
    )
    # ax[1].semilogy(post.locations[:-1], sigmas)
    # ax[1].semilogy(
    #     candidates[np.linalg.norm(errors, axis=1) > np.median(np.linalg.norm(errors, axis=1))],
    #     np.linalg.norm(errors, axis=1)[np.linalg.norm(errors, axis=1) > np.median(np.linalg.norm(errors, axis=1))],
    #     "+",
    #     color="green",
    #     label="Error",
    # )

    # ax[1].semilogy(
    #     evalgrid, discrepancy, linestyle="dashed", color="steelblue", label="True Error"
    # )
    # ax[1].semilogy(
    #     evalgrid,
    #     scipy_discrepancy,
    #     color="gray",
    #     linestyle="dotted",
    #     label="Scipy error",
    # )
    ax[1].axhline(1, color="black")
    ax[1].axhline(100, color="black")

    ax[2].semilogy(post.locations[:-1], np.diff(post.locations), color="k", alpha=0.8)
    ax[2].semilogy(refsol.x[:-1], np.diff(refsol.x), color="steelblue")
    # ax[0].set_ylim((-1.5, 3.5))
    ax[1].set_ylim((1e-5, 1e8))
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
