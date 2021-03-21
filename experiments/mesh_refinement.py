"""Compute the solution to the bratus BVP with a probabilistic solver.




http://www.orcca.on.ca/TechReports/TechReports/2001/TR-01-02.pdf


"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars, filtsmooth, diffeq
from probnum._randomvariablelist import _RandomVariableList
from bvps import (
    r_example,
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
)
from tqdm import tqdm
import pandas as pd


from probnum import random_variables as randvars


from probnumeval.timeseries import (
    average_normalized_estimation_error_squared,
    root_mean_square_error,
    non_credibility_index,
)


from scipy.integrate import solve_bvp


bvp = r_example(xi=0.01)

initial_grid = np.linspace(bvp.t0, bvp.tmax, 5)
initial_guess = np.zeros((2, len(initial_grid)))
refsol = solve_bvp(bvp.f, bvp.scipy_bc, initial_grid, initial_guess, tol=1e-15)


q = 3
num_gridpoints = 50

ibm = statespace.IBM(
    ordint=q,
    spatialdim=2,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)

integ = WrappedIntegrator(ibm, bvp)


rv = randvars.Normal(np.ones(ibm.dimension), np.eye(ibm.dimension))
initrv, _ = integ.forward_rv(rv, t=bvp.t0, dt=0.0)

measmod = from_ode(bvp, ibm)
# measmod = filtsmooth.IteratedDiscreteComponent(measmod)
# stopcrit = MyStoppingCriterion(atol=1e-2, rtol=1e-2, maxit=500)
# stopcrit = MyStoppingCriterion()

# measmod_iterated = filtsmooth.IteratedDiscreteComponent(measmod, stopcrit=stopcrit)
kalman = MyKalman(dynamics_model=integ, measurement_model=measmod, initrv=initrv)

# kalman_iterated = filtsmooth.Kalman(
#     dynamics_model=integ, measurement_model=measmod_iterated, initrv=initrv
# )
P0 = ibm.proj2coord(0)
evalgrid = np.sort(np.random.rand(234))

num_gridpoints = 2
grid = np.linspace(bvp.t0, bvp.tmax, num_gridpoints)
data = np.zeros((len(grid), 2))

#
#
#
#
#
#
#
#
#
#

# out_ks_ekfA = diffeq.KalmanODESolution(out_ks_ekf2)
# out_ks_ekfX = kalman.filtsmooth(
#     dataset=data, times=grid, _previous_posterior=out_ks_ekf2
# )
# print(out_ks_ekf.y.mean)
# plt.plot(out_ks_ekf.t, out_ks_ekf.y.mean[:, 0])
# plt.plot(out_ks_ekf.t, out_ks_ekfA.y.mean[:, 0])
# plt.show()


###### Next round #######
evalgrid = np.linspace(bvp.t0, bvp.tmax, 100)
old_posterior = None
for i in range(10):
    stopcrit = MyStoppingCriterion(atol=1e-6, rtol=1e-6)
    old_posterior = kalman.iterated_filtsmooth(
        dataset=data, times=grid, stopcrit=stopcrit, old_posterior=old_posterior
    )

    out_ieks_ekf = diffeq.KalmanODESolution(old_posterior)
    print("NUMBER OF ITERATIONS", stopcrit.previous_number_of_iterations)
    out_ieks_ekf_ssq = kalman.ssq

    plt.plot(evalgrid, out_ieks_ekf(evalgrid).mean[:, 0])
    # plt.plot(evalgrid, refsol.sol(evalgrid).T[:, 0], linestyle="dashed")
    # plt.plot(out_ieks_ekf.t, out_ieks_ekf.y.mean[:, 0])
    for t in out_ieks_ekf.t:
        plt.axvline(t, linewidth=0.2)
    plt.show()

    new_grid_ = new_grid2(out_ieks_ekf.t)
    errors = out_ieks_ekf(new_grid_).std * np.sqrt(out_ieks_ekf_ssq)

    msrvs = _RandomVariableList(
        [measmod.forward_rv(old_posterior(t), t=t)[0] for t in new_grid_]
    )
    errors = msrvs.std * np.sqrt(out_ieks_ekf_ssq)
    # print(errors, error2)
    new_t = new_grid_[np.linalg.norm(errors, axis=1) > np.mean(np.abs(errors))]
    # new_t = new_grid_[np.linalg.norm(errors, axis=1) > 1e-1]
    grid = np.sort(np.append(grid, new_t))
    print("ERROR MAGNITUDE", np.linalg.norm(errors, axis=1))
    # grid = new_t
    data = np.zeros((len(grid), 2))
    print("MEAN ERROR", np.mean(np.abs(errors)))
    print("NUMBER OF GRIDPOINTS", len(grid))
    print()


###### Next round #######


# out_ieks_ekf = diffeq.KalmanODESolution(
#     kalman.iterated_filtsmooth(dataset=data, times=grid, stopcrit=stopcrit)
# )
# out_ieks_ekf_ssq = kalman.ssq

# plt.plot(out_ieks_ekf.t, out_ieks_ekf.y.mean[:, 0])
# for t in out_ieks_ekf.t:
#     plt.axvline(t, linewidth=0.2)
# plt.show()


# new_grid_ = new_grid2(out_ieks_ekf.t)
# errors = out_ieks_ekf(new_grid_).std * np.sqrt(out_ieks_ekf_ssq)
# print(errors)

# new_t = new_grid_[np.linalg.norm(errors, axis=1) > np.mean(np.abs(errors))]
# grid = np.sort(np.append(grid, new_t))

# # grid = new_t
# data = np.zeros((len(grid), 2))


# ###### Next round #######

# out_ieks_ekf = diffeq.KalmanODESolution(
#     kalman.iterated_filtsmooth(dataset=data, times=grid, stopcrit=stopcrit)
# )
# out_ieks_ekf_ssq = kalman.ssq

# plt.plot(out_ieks_ekf.t, out_ieks_ekf.y.mean[:, 0])
# for t in out_ieks_ekf.t:
#     plt.axvline(t, linewidth=0.2)
# plt.show()


# print(new_t.size, grid.size)
# print(errors)
# # plt.plot(out.locations, out.state_rvs.mean[:, 0])
# # plt.show()