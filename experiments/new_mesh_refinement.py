"""Try out probsolve_bvp."""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars, filtsmooth, diffeq
from probnum._randomvariablelist import _RandomVariableList
from bvps import (
    r_example,
    bratus,
    BoundaryValueProblem,
    WrappedIntegrator,
    from_ode,
    MyKalman,
    generate_samples,
    split_grid,
    new_grid,
    new_grid2,
    matlab_example,
    MyStoppingCriterion,
    MyIteratedDiscreteComponent,
    probsolve_bvp,bratus_second_order,
    matlab_example_second_order
)
from tqdm import tqdm
import pandas as pd


from probnum import random_variables as randvars


from scipy.integrate import solve_bvp


# bvp = r_example(xi=0.01)
# # bvp = matlab_example()

# bvp = bratus_second_order()
# bvp1st = bratus()



bvp = matlab_example_second_order()
bvp1st = matlab_example()

initial_grid = np.linspace(bvp.t0, bvp.tmax, 100)
initial_guess = np.zeros((2, len(initial_grid)))
refsol = solve_bvp(bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=1e-12)


q = 4

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
    atol=1e-14,
    rtol=1e-14,
    insert="single",
    which_method="iekf",
    maxit=5
)


evalgrid = np.linspace(bvp.t0, bvp.tmax)

for post, ssq, errors in posterior:

    fig, ax = plt.subplots(nrows=3, sharex=True, dpi=400)
    m = post(evalgrid).mean[:, 0]
    s = post(evalgrid).std[:, 0] * np.sqrt(ssq)


    discrepancy = np.abs(refsol.sol(evalgrid).T[:, 0] - m)
    ax[0].plot(evalgrid, m)
    ax[0].fill_between(evalgrid, m - 3*s, m + 3*s, alpha=0.1)
    ax[0].plot(evalgrid, refsol.sol(evalgrid).T[:, 0], color="gray", linestyle="dashed")
    for t in post.locations:
        ax[0].axvline(t, linewidth=0.1, color="k")  #


    ax[1].semilogy(evalgrid, s)
    ax[1].semilogy(evalgrid, discrepancy)
    ax[2].semilogy(post.locations[:-1], np.diff(post.locations), "x", color="k")


    ax[0].set_ylim((-1.5, 1.5))
    ax[1].set_ylim((1e-14, 1e1))
    ax[2].set_ylim((1e-4, 1e0))

    ax[2].set_xlabel("Time")
    ax[0].set_ylabel("Solution")
    ax[1].set_ylabel("Error/Estimation")
    ax[2].set_ylabel("Stepsize")
    plt.show()