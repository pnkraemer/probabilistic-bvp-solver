"""This script chisels out the advantage of using a bridge prior over not using one."""

from bvps import bvp_solver, problem_examples, quadrature
import numpy as np

import pytest
from probnum import statespace, filtsmooth, randvars
import matplotlib.pyplot as plt
from tqdm import tqdm


# bvp = problem_examples.problem_7_second_order(xi=1e-2)
bvp = problem_examples.problem_20_second_order(xi=1e-2)
ibm = statespace.IBM(
    ordint=2,
    spatialdim=bvp.dimension,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
N = 10
initial_grid = np.linspace(bvp.t0, bvp.tmax, N)
t = np.linspace(bvp.t0, bvp.tmax, 200)

MAXIT = 3

initial_guess = np.ones((N, bvp.dimension))


fig, ax = plt.subplots(
    ncols=2,
    figsize=(5.5, 2.5),
    nrows=ibm.ordint + 1,
    sharex=True,
    sharey="row",
    dpi=200,
    gridspec_kw={"height_ratios": [4, 1, 1]},
    constrained_layout=True,
)
for USE_BRIDGE, axis in zip([True, False], ax.T):
    solver = bvp_solver.BVPSolver.from_default_values(
        ibm, use_bridge=USE_BRIDGE, initial_sigma_squared=1e12
    )
    solution_gen = solver.solution_generator(
        bvp,
        atol=1e-0,
        rtol=1e-0,
        initial_grid=initial_grid,
        initial_guess=initial_guess,
        maxit_ieks=MAXIT,
        yield_ieks_iterations=True,
    )

    # Only the first 20 iterations, i.e. the first 4 refinements (since maxit_ieks=5)
    for i, (kalman_posterior, sigma_squared) in zip(tqdm(range(MAXIT)), solution_gen):

        y = kalman_posterior(t).mean
        s = kalman_posterior(t).std * np.sqrt(sigma_squared)
        for q, curr_ax in zip(range(ibm.ordint + 1), axis):
            curr_ax.plot(t, y[:, q], color="k", alpha=0.3 + 0.7 * float(i / (MAXIT)))
            # curr_ax.fill_between(t, y[:, q] - 2 * s[:, q], y[:, q] + 2*s[:, q], color="k", alpha=0.1)
            #
            curr_ax.plot(
                kalman_posterior.locations,
                kalman_posterior.states.mean[:, q],
                ".",
                color="black",
                alpha=0.3 + 0.7 * float(i / (MAXIT)),
            )
    for q, curr_ax in zip(range(ibm.ordint + 1), axis):
        if q == 0:
            curr_ax.plot(t, bvp.solution(t), color="salmon", linestyle="dashed")

        # for x in kalman_posterior.locations:
        #    curr_ax.axvline(x, linewidth=0.25, color="k")


ax[0][0].set_title("Bridge")
ax[0][1].set_title("No Bridge")
ax[0][0].set_ylabel("State y")
ax[1][0].set_ylabel("dy")
ax[2][0].set_ylabel("ddy")
plt.savefig("bridge_advantage.pdf")
plt.show()
