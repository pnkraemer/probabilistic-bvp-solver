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

TOL = 1e-2

# bvp = r_example(xi=0.01)
# # bvp = matlab_example()

TMAX = 0.2
XI = 0.001
bvp = problem_7_second_order(xi=XI)
bvp1st = problem_7(xi=XI)


bvp = bratus_second_order()
# bvp1st = bratus()

# bvp = matlab_example_second_order(tmax=TMAX)
# bvp1st = matlab_example(tmax=TMAX)

print(bvp1st.y0, bvp1st.ymax)
print(bvp1st.L, bvp1st.R)


# bvp = problem_7_second_order(xi=0.1)
# bvp1st = problem_7(xi=0.1)

# initial_grid = np.union1d(
#     np.linspace(bvp.t0, 0.3, 100), np.linspace(bvp.t0, bvp.tmax, 100)
# )
initial_grid = np.linspace(bvp.t0, bvp.tmax, 25)
initial_guess = np.zeros((2, len(initial_grid)))
refsol = solve_bvp(bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=TOL)
refsol_fine = solve_bvp(
    bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=1e-12
)
bvp.solution = refsol_fine.sol
# initial_grid = refsol.x.copy()

# initial_grid = refsol.x

# plt.plot(refsol.x, np.ones_like(refsol.x), ".")
# plt.show()
# print(refsol.x)
# assert False

q = 5


ibm = statespace.IBM(
    ordint=q,
    spatialdim=2,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
# ibm.equivalent_discretisation_preconditioned._proc_noise_cov_cholesky *= 1e5

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
    maxit=115,
    ignore_bridge=False,
    which_errors="probabilistic_defect",
    refinement="tolerance",
)


print(
    "Current information: EM in the filter sucks, EM on iterated filtsmooth level is okay, no EM is fine too."
)


evalgrid = np.linspace(bvp.t0, bvp.tmax, 150, endpoint=True)

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
) in enumerate(posterior):

    # print(post.locations[1:][insert_one])
    # print(
    #     "Why is the filtering posterior soooo bad even if the smoothing posterior is alright?"
    # )
    post2 = post.filtering_solution
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
    m_ = evaluated.mean[:, :2]
    # s = evaluated.std[:, :2] * np.sqrt(ssq)

    # evaluated2 = post2(evalgrid)
    # m2 = evaluated2.mean
    # s2 = evaluated2.std * np.sqrt(ssq)

    t = post.locations
    m = post.states.mean
    s = post.states.std

    t2 = post2.locations
    m2 = post2.states.mean
    s2 = post2.states.std
    # print(m, m2)

    # discrepancy = np.abs(refsol.sol(evalgrid).T[:, 0] - post(evalgrid).mean[:, 0])
    ax[0].plot(t, m, color="k")
    ax[0].plot(t2, m2, color="orange")
    ax[0].plot(
        evalgrid, refsol_fine.sol(evalgrid).T, color="steelblue", linestyle="dashed"
    )
    # ax[0].plot(
    #     evalgrid, bvp.solution(evalgrid).T[:, 0], color="red", linestyle="dotted"
    # )

    # print(s)
    # print(s2)

    # ax[0].fill_between(t, m - 3 * s, m + 3 * s)
    # ax[0].fill_between(t2, m2 - 3 * s2, m2 + 3 * s2)

    for t in post.locations:
        ax[0].axvline(t, linewidth=0.1, color="k")

    discrepancy_ = np.abs(bvp.solution(evalgrid).T - post(evalgrid).mean[:, :2])

    # print(bvp.solution(evalgrid).T[0])
    # print(post(evalgrid).mean[:, :2][0])
    # print(discrepancy[0])

    discrepancy = discrepancy_ / (TOL * (1.0 + np.abs(m_)))
    # print(
    #     discrepancy[0],
    #     post(evalgrid).mean[:, :2][0],
    #     (TOL * (1.0 + np.abs(post(evalgrid).mean[:, :2])))[0],
    # )
    discrepancy = np.linalg.norm(discrepancy, axis=1)
    # print(discrepancy[0])
    # scipy_discrepancy = np.abs(refsol.sol(evalgrid)[0] - bvp.solution(evalgrid)[0])

    # ax[1].semilogy(evalgrid, s, color="k", label="Uncertainty")
    ax[1].semilogy(
        post.locations[:-1] + 0.5 * np.diff(post.locations),
        integral_error,
        ".",
        color="black",
        label="Estimated",
    )
    ax[1].axhspan(0.0, 1.0, color="C0", alpha=0.2)
    ax[1].axhspan(1.0, 3.0 ** (q + 0.5), color="C1", alpha=0.2)
    ax[1].axhspan(3.0 ** (q + 0.5), 10000000000000, color="C2", alpha=0.2)
    # ax[1].semilogy(post.locations[:-1], sigmas)
    # ax[1].semilogy(
    #     candidates[np.linalg.norm(errors, axis=1) > np.median(np.linalg.norm(errors, axis=1))],
    #     np.linalg.norm(errors, axis=1)[np.linalg.norm(errors, axis=1) > np.median(np.linalg.norm(errors, axis=1))],
    #     "+",
    #     color="green",
    #     label="Error",
    # )

    ax[1].semilogy(
        evalgrid,
        discrepancy,
        linestyle="dashdot",
        color="black",
        label="True quotient",
    )
    ax[1].semilogy(
        evalgrid,
        np.linalg.norm(discrepancy_, axis=1),
        linestyle="dashed",
        color="black",
        label="True Error",
    )
    # ax[1].semilogy(
    #     evalgrid,
    #     scipy_discrepancy,
    #     color="gray",
    #     linestyle="dotted",
    #     label="Scipy error",
    # )
    ax[1].axhline(1, color="black")
    ax[1].axhline(3.0 ** (q + 0.5), color="black")

    ax[2].semilogy(post.locations[:-1], np.diff(post.locations), color="k", alpha=0.8)
    ax[2].semilogy(refsol.x[:-1], np.diff(refsol.x), color="steelblue")
    # ax[0].set_ylim((-112.5, 113.5))
    ax[1].set_ylim((1e-5, 1e8))
    ax[2].set_ylim((1e-4, 1e0))
    ax[1].legend(frameon=False)

    ax[2].set_xlabel("Time")
    ax[0].set_ylabel("Solution")
    ax[1].set_ylabel("Error ratio")
    ax[2].set_ylabel("Stepsize")
    ax[0].set_title(
        f"Refinement {idx + 1}: $N={len(post.locations)}$ Points | Scipy {len(refsol.x)}"
    )
    fig.align_ylabels()
    plt.show()

    # print(post.kalman_posterior.states[0].mean)
    # print(post.kalman_posterior.states[1].mean)
    # print(post.kalman_posterior.filtering_posterior.states[0].mean)
    # print()
    # print(post.kalman_posterior.states[0].std)
    # print(post.kalman_posterior.states[1].std)
    # print(post.kalman_posterior.filtering_posterior.states[0].std)
