"""Compute the solution to the bratus BVP with a probabilistic solver."""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars, filtsmooth, diffeq
from bvps import (
    r_example,
    BoundaryValueProblem,
    WrappedIntegrator,
    from_ode,
    MyKalman,
    generate_samples,
)
from tqdm import tqdm
import pandas as pd


from probnum import random_variables as randvars


from probnumeval.timeseries import (
    average_normalized_estimation_error_squared,
    root_mean_square_error,
    non_credibility_index,
)


def rmse(kalpost, refsol, locs):

    f = lambda x: kalpost(x).mean
    g = lambda x: refsol(x).T
    return root_mean_square_error(f, g, locs)


def anees(kalpost, refsol, locs):
    g = lambda x: refsol(x).T
    return average_normalized_estimation_error_squared(kalpost, g, locs)


def nci(kalpost, refsol, locs):
    g = lambda x: refsol(x).T
    return non_credibility_index(kalpost, g, locs)


def dataframe(row_labels, column_labels):
    data = np.zeros((len(row_labels), len(column_labels)))
    return pd.DataFrame(data=data, index=row_labels, columns=column_labels)


from scipy.integrate import solve_bvp


bvp = r_example(xi=0.1)

print(bvp.y0)

initial_grid = np.linspace(bvp.t0, bvp.tmax, 500)
initial_guess = np.zeros((2, len(initial_grid)))
refsol = solve_bvp(bvp.f, bvp.scipy_bc, initial_grid, initial_guess, tol=1e-15)


q = 4
num_gridpoints = 50

ibm = statespace.IBM(
    ordint=q,
    spatialdim=2,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)

integ = WrappedIntegrator(ibm, bvp)

print()


rv = randvars.Normal(np.ones(ibm.dimension), np.eye(ibm.dimension))
initrv, _ = integ.forward_rv(rv, t=bvp.t0, dt=0.0)

measmod = from_ode(bvp, ibm)

stopcrit = filtsmooth.StoppingCriterion(atol=1e-1, rtol=1e-1, maxit=500)

# measmod_iterated = filtsmooth.IteratedDiscreteComponent(measmod, stopcrit=stopcrit)
kalman = MyKalman(dynamics_model=integ, measurement_model=measmod, initrv=initrv)

# kalman_iterated = filtsmooth.Kalman(
#     dynamics_model=integ, measurement_model=measmod_iterated, initrv=initrv
# )
P0 = ibm.proj2coord(0)
evalgrid = np.sort(np.random.rand(234))


labels = ["IEKS-EKF", "KS-EKF"]

gridpoint_set = 2 ** np.arange(1, 9)
results_rmse = dataframe(row_labels=gridpoint_set, column_labels=labels)
results_anees = dataframe(row_labels=gridpoint_set, column_labels=labels)
results_nci = dataframe(row_labels=gridpoint_set, column_labels=labels)

for num_gridpoints in tqdm(gridpoint_set):

    grid = np.linspace(bvp.t0, bvp.tmax, num_gridpoints)
    data = np.zeros((len(grid), 2))

    smp = np.random.randn(len(rv.mean), len(grid)).T
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

    out_ks_ekf2 = kalman.filtsmooth(dataset=data, times=grid)
    # out_ks_ekfA = diffeq.KalmanODESolution(out_ks_ekf2)
    # out_ks_ekfX = kalman.filtsmooth(
    #     dataset=data, times=grid, _previous_posterior=out_ks_ekf2
    # )
    out_ks_ekf_ssq = kalman.ssq
    out_ks_ekf = diffeq.KalmanODESolution(out_ks_ekf2)
    # print(out_ks_ekf.y.mean)
    # plt.plot(out_ks_ekf.t, out_ks_ekf.y.mean[:, 0])
    # plt.plot(out_ks_ekf.t, out_ks_ekfA.y.mean[:, 0])
    # plt.show()
    out_ieks_ekf = diffeq.KalmanODESolution(
        kalman.iterated_filtsmooth(dataset=data, times=grid, stopcrit=stopcrit)
    )
    out_ieks_ekf_ssq = kalman.ssq

    # out_ieks_iekf = diffeq.KalmanODESolution(
    #     kalman_iterated.iterated_filtsmooth(dataset=data, times=grid, stopcrit=stopcrit)
    # )
    # out_ks_iekf = diffeq.KalmanODESolution(
    #     kalman_iterated.filtsmooth(dataset=data, times=grid)
    # )

    # x1 = (refsol.sol(evalgrid)).T
    # x_ieks_ekf = (out_ieks_ekf(evalgrid).mean) @ P0.T
    # x_ks_ekf = (out_ks_ekf(evalgrid).mean) @ P0.T
    # x_ieks_iekf = (out_ieks_iekf(evalgrid).mean) @ P0.T
    # x_ks_iekf = (out_ks_iekf(evalgrid).mean) @ P0.T

    # RMSE
    results_rmse["IEKS-EKF"][num_gridpoints] = rmse(out_ieks_ekf, refsol.sol, evalgrid)
    # results_rmse["IEKS-IEKF"][num_gridpoints] = rmse(
    #     out_ieks_iekf, refsol.sol, evalgrid
    # )
    results_rmse["KS-EKF"][num_gridpoints] = rmse(out_ks_ekf, refsol.sol, evalgrid)
    # results_rmse["KS-IEKF"][num_gridpoints] = rmse(out_ks_iekf, refsol.sol, evalgrid)

    # ANEES
    results_anees["IEKS-EKF"][num_gridpoints] = (
        anees(out_ieks_ekf, refsol.sol, evalgrid) / out_ieks_ekf_ssq
    )
    # results_anees["IEKS-IEKF"][num_gridpoints] = anees(
    #     out_ieks_iekf, refsol.sol, evalgrid
    # )
    results_anees["KS-EKF"][num_gridpoints] = (
        anees(out_ks_ekf, refsol.sol, evalgrid) / out_ks_ekf_ssq
    )
    # results_anees["KS-IEKF"][num_gridpoints] = anees(out_ks_iekf, refsol.sol, evalgrid)

    # NCI
    results_nci["IEKS-EKF"][num_gridpoints] = nci(
        out_ieks_ekf, refsol.sol, evalgrid
    ) + 10 * np.log10(1.0 / out_ieks_ekf_ssq)
    # results_nci["IEKS-IEKF"][num_gridpoints] = nci(out_ieks_iekf, refsol.sol, evalgrid)
    results_nci["KS-EKF"][num_gridpoints] = nci(
        out_ks_ekf, refsol.sol, evalgrid
    ) + 10 * np.log10(1.0 / out_ks_ekf_ssq)
    # results_nci["KS-IEKF"][num_gridpoints] = nci(out_ks_iekf, refsol.sol, evalgrid)

results_rmse.to_csv("data/workprecision_first_attempt_r_example_rmse.csv")
results_anees.to_csv("data/workprecision_first_attempt_r_example_anees.csv")
results_nci.to_csv("data/workprecision_first_attempt_r_example_nci.csv")

print(results_rmse)
print(results_anees)
print(results_nci)
# plt.plot(out.locations, out.state_rvs.mean[:, 0])
# plt.show()
