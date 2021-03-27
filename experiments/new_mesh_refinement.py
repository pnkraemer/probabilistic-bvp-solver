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
    matlab_example_second_order, problem_7_second_order, problem_7
)
from tqdm import tqdm
import pandas as pd


from probnum import random_variables as randvars


from scipy.integrate import solve_bvp


# bvp = r_example(xi=0.01)
# # bvp = matlab_example()

# bvp = bratus_second_order()
# bvp1st = bratus()



bvp = matlab_example_second_order(tmax=1)
bvp1st = matlab_example(tmax=1)





# bvp = problem_7_second_order(xi=0.1)
# bvp1st = problem_7(xi=0.1)

initial_grid = np.linspace(bvp.t0, bvp.tmax, 10)
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
    insert="triple",
    which_method="iekf",
    maxit=10
)


evalgrid = np.linspace(bvp.t0, bvp.tmax, 300)

for idx, (post, ssq, errors, kalpost, candidates) in enumerate(posterior):

    fig, ax = plt.subplots(nrows=3, sharex=True, dpi=400)
    evaluated = post(evalgrid)
    m = evaluated.mean[:, 0]
    s = evaluated.std[:, 0] * np.sqrt(ssq)

    discrepancy = np.abs(refsol.sol(evalgrid).T[:, 0] - m)
    ax[0].plot(evalgrid, evaluated.mean, color="k")
    ax[0].plot(evalgrid, refsol.sol(evalgrid).T[:, 0], color="steelblue", linestyle="dashed")


    discrepancy = np.abs(bvp.solution(evalgrid)[0] - m)
    scipy_discrepancy = np.abs(refsol.sol(evalgrid)[0] - bvp.solution(evalgrid)[0])

    ax[1].semilogy(evalgrid, s, color="k", label="Uncertainty")
    ax[1].semilogy(candidates, np.linalg.norm(errors, axis=1), "+", color="darksalmon", label="Error")
    ax[1].semilogy(evalgrid, discrepancy, color="steelblue", linestyle="dashed", label="Truth")
    ax[1].semilogy(evalgrid, scipy_discrepancy, color="gray", linestyle="dotted", label="scipy error")




    ax[2].semilogy(post.locations[:-1], np.diff(post.locations), "x", color="k")
    ax[2].semilogy(refsol.x[:-1], np.diff(refsol.x), ".", color="steelblue")

    # ax[0].set_ylim((-1.5, 3.5))
    ax[1].set_ylim((1e-14, 1e8))
    ax[2].set_ylim((1e-4, 1e0))
    ax[1].legend(frameon=False)

    ax[2].set_xlabel("Time")
    ax[0].set_ylabel("Solution")
    ax[1].set_ylabel("Error & Estimation")
    ax[2].set_ylabel("Stepsize")
    ax[0].set_title(f"Refinement #{idx + 1}: $N={len(post.locations)}$ Points | Scipy {len(refsol.x)}")
    fig.align_ylabels()
    plt.show()