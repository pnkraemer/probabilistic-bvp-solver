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
    probsolve_bvp,
)
from tqdm import tqdm
import pandas as pd


from probnum import random_variables as randvars


from scipy.integrate import solve_bvp


bvp = r_example(xi=0.001)
# bvp = matlab_example()

initial_grid = np.linspace(bvp.t0, bvp.tmax, 10)
initial_guess = np.zeros((2, len(initial_grid)))
refsol = solve_bvp(bvp.f, bvp.scipy_bc, initial_grid, initial_guess, tol=1e-12)


q = 3

ibm = statespace.IBM(
    ordint=q,
    spatialdim=2,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)

integ = WrappedIntegrator(ibm, bvp)


posterior = probsolve_bvp(
    bvp=bvp,
    bridge_prior=integ,
    initial_grid=initial_grid,
    atol=1e-4,
    rtol=1e-4,
    insert="single",
    which_method="iekf",
)


evalgrid = np.linspace(bvp.t0, bvp.tmax)

for post in posterior:
    fig, ax = plt.subplots(nrows=2)

    ax[0].plot(evalgrid, post(evalgrid).mean[:, 0])
    ax[0].plot(evalgrid, refsol.sol(evalgrid).T[:, 0], color="gray", linestyle="dashed")
    for t in post.locations:
        ax[0].axvline(t, linewidth=0.1, color="k")  #
    ax[1].semilogy(post.locations[:-1], np.diff(post.locations), "x", color="k")
    plt.show()