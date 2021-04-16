"""Solving BVPs."""
import numpy as np
from probnum import diffeq, randvars, utils
from probnum._randomvariablelist import _RandomVariableList

from ._mesh import *
from ._ode_measmods import from_ode, from_second_order_ode
from ._problems import SecondOrderBoundaryValueProblem
from ._kalman import (
    ConstantStopping,
    MyIteratedDiscreteComponent,
    MyKalman,
    MyStoppingCriterion,
)

from ._bvp_initialise import *
from ._integrators import WrappedIntegrator
from ._control import *

import scipy.linalg
from ._error_estimates import *


def refine_median(quotient):
    return quotient > np.median(quotient)


def refine_tolerance(quotient):
    return quotient > 1


REFINEMENT_OPTIONS = {"median": refine_median, "tolerance": refine_tolerance}


CANDIDATE_LOCATIONS_OPTIONS = {
    "single": insert_central_point,
    "double": insert_two_equispaced_points,
    "triple": None,
}


LARGE_VALUE = 1e5


def probsolve_bvp(
    bvp,
    bridge_prior,
    initial_grid,
    atol,
    rtol,
    maxit=50,
    which_method="iekf",
    insert="double",
    which_errors="defect",
    ignore_bridge=False,
    refinement="median",
):
    """Solve a BVP.

    Parameters
    ----------
    bridge_prior : WrappedIntegrator
    which_method : either "ekf" or "iekf".
    insert : "single" or "double"
    """
    ibm = bridge_prior.integrator
    ibm.equivalent_discretisation_preconditioned._proc_noise_cov_cholesky *= np.sqrt(
        LARGE_VALUE
    )

    bridge_prior = WrappedIntegrator(ibm, bvp)

    refinement_function = REFINEMENT_OPTIONS[refinement]
    estimate_errors_function = ESTIMATE_ERRORS_OPTIONS[which_errors]
    candidate_function = CANDIDATE_LOCATIONS_OPTIONS[insert]
    # Construct Kalman object ##############################

    if isinstance(bvp, SecondOrderBoundaryValueProblem):
        print("Recognised a 2nd order BVP")
        measmod = from_second_order_ode(bvp, bridge_prior)

        bvp_dim = len(bvp.R.T) // 2

    else:
        measmod = from_ode(bvp, bridge_prior)
        bvp_dim = len(bvp.R.T)

    if which_method == "iekf":
        stopcrit_iekf = ConstantStopping(maxit=maxit)
        # stopcrit_iekf = MyStoppingCriterion(
        #     atol=100 * atol, rtol=100 * rtol, maxit=maxit
        # )
        measmod = MyIteratedDiscreteComponent(measmod, stopcrit=stopcrit_iekf)

    rv = randvars.Normal(
        0 * np.ones(bridge_prior.dimension),
        LARGE_VALUE * np.eye(bridge_prior.dimension),
        cov_cholesky=np.sqrt(LARGE_VALUE) * np.eye(bridge_prior.dimension),
    )
    initrv, _ = bridge_prior.forward_rv(rv, t=bvp.t0, dt=0.0)

    if ignore_bridge:
        print("No bridge detected.")
        kalman = MyKalman(
            dynamics_model=bridge_prior.integrator, measurement_model=measmod, initrv=rv
        )
    else:
        kalman = MyKalman(
            dynamics_model=bridge_prior, measurement_model=measmod, initrv=initrv
        )

    stopcrit_bvp = MyStoppingCriterion(atol=atol, rtol=rtol, maxit=maxit)
    # stopcrit_bvp = ConstantStopping(maxit=maxit)

    # Call IEKS ##############################

    grid = initial_grid
    # stopcrit_ieks = MyStoppingCriterion(atol=atol, rtol=rtol, maxit=maxit)
    stopcrit_ieks = ConstantStopping(maxit=maxit)

    # # Initial solve
    # data = np.zeros((len(grid), bvp_dim))
    # if ignore_bridge:
    #     kalman_posterior = kalman.iterated_filtsmooth(
    #         dataset=data,
    #         times=grid,
    #         stopcrit=stopcrit_ieks,
    #         measmodL=bridge_prior.measmod_L,
    #         measmodR=bridge_prior.measmod_R,
    #     )
    # else:
    #     kalman_posterior = kalman.iterated_filtsmooth(
    #         dataset=data,
    #         times=grid,
    #         stopcrit=stopcrit_ieks,
    #         measmodL=None,
    #         measmodR=None,
    #     )

    kalman_posterior = bvp_initialise_ode(
        bvp=bvp,
        bridge_prior=bridge_prior,
        initial_grid=initial_grid,
    )

    bvp_posterior = diffeq.KalmanODESolution(kalman_posterior)
    # sigma_squared = kalman.ssq
    sigma_squared = 1.0
    sigmas = np.ones_like(bvp_posterior.locations[:-1])
    # kalman_posterior = bvp_posterior.kalman_posterior

    # Set up candidates for mesh refinement

    (
        new_mesh,
        integral_error,
        quotient,
        candidate_locations,
        h,
        insert_one,
        insert_two,
    ) = control(
        bvp_posterior,
        kalman_posterior,
        sigma_squared,
        measmod,
        atol,
        rtol,
    )
    errors = None
    # candidate_locations, h = candidate_function(bvp_posterior.locations)
    print(
        f"Next: go from {len(bvp_posterior.locations)} points to {len(new_mesh)} points."
    )
    print(f"SSQ={sigma_squared}")
    # # Estimate errors and choose nodes to refine
    # errors, reference, quotient = estimate_errors_function(
    #     bvp_posterior,
    #     kalman_posterior,
    #     candidate_locations,
    #     sigma_squared,
    #     measmod,
    #     atol,
    #     rtol,
    # )
    # # errors *= h[:, None]

    # print(errors.shape, h.shape)

    # magnitude = stopcrit_bvp.evaluate_error(error=errors, reference=reference)
    # quotient = stopcrit_bvp.evaluate_quotient(errors, reference)
    # quotient = np.linalg.norm(quotient, axis=1)
    yield bvp_posterior, sigma_squared, integral_error, kalman_posterior, candidate_locations, h, quotient, sigmas, insert_one, insert_two

    # print(quotient.shape)
    mask = refinement_function(quotient)
    # print(mask)
    # print(quotient)
    # print(quotient)
    # if refinement == "median":
    #     mask = refine_median(quotient)
    # else:
    #     mask = refine_tolerance(quotient)
    while np.any(mask):
        # while True:

        sigma = np.sqrt(sigma_squared)

        ibm = bridge_prior.integrator
        ibm.equivalent_discretisation_preconditioned._proc_noise_cov_cholesky *= (
            np.sqrt(sigma)
        )
        ibm.equivalent_discretisation_preconditioned.proc_noise_cov_mat *= sigma

        bridge_prior = WrappedIntegrator(ibm, bvp)

        new_initrv = kalman_posterior.states[0]
        new_mean = new_initrv.mean.copy()
        new_cov_cholesky = utils.linalg.cholesky_update(
            sigma * new_initrv.cov_cholesky, new_mean - kalman.initrv.mean
        )
        # new_cov_cholesky = kalman.initrv.cov_cholesky
        new_cov = new_cov_cholesky @ new_cov_cholesky.T
        kalman.initrv = randvars.Normal(
            mean=new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky
        )

        # Refine grid
        # print(mask.shape)
        # new_points = candidate_locations[mask]

        # new_fullgrid = np.union1d(grid, new_points)
        # sparse_fullgrid = np.union1d(new_fullgrid, new_fullgrid[:-1] + 0.5*np.diff(new_fullgrid))[::3]
        # grid = np.union1d(sparse_fullgrid, grid[[0, -1]])

        # grid = np.union1d(grid, new_mesh)
        # print(grid.shape, new_points.shape, candidate_locations.shape)

        data = np.zeros((len(new_mesh), bvp_dim))
        # Compute new solution
        if ignore_bridge:
            kalman_posterior = kalman.iterated_filtsmooth(
                dataset=data,
                times=new_mesh,
                stopcrit=stopcrit_ieks,
                measmodL=bridge_prior.measmod_L,
                measmodR=bridge_prior.measmod_R,
                old_posterior=kalman_posterior,
            )
        else:
            kalman_posterior = kalman.iterated_filtsmooth(
                dataset=data,
                times=new_mesh,
                stopcrit=stopcrit_ieks,
                measmodL=None,
                measmodR=None,
                old_posterior=kalman_posterior,
            )
        bvp_posterior = diffeq.KalmanODESolution(kalman_posterior)
        sigma_squared = kalman.ssq
        sigmas = kalman.sigmas

        (
            new_mesh,
            integral_error,
            quotient,
            candidate_locations,
            h,
            insert_one,
            insert_two,
        ) = control(
            bvp_posterior,
            kalman_posterior,
            sigma_squared,
            measmod,
            atol,
            rtol,
        )
        errors = None
        print(
            f"Next: go from {len(bvp_posterior.locations)} points to {len(new_mesh)} points."
        )
        print(f"SSQ={sigma_squared}")
        # print(new_mesh)

        # # Set up candidates for mesh refinement
        # candidate_locations, h = candidate_function(bvp_posterior.locations)

        # # Estimate errors and choose nodes to refine
        # errors, reference, quotient = estimate_errors_function(
        #     bvp_posterior,
        #     kalman_posterior,
        #     candidate_locations,
        #     sigma_squared,
        #     measmod,
        #     atol,
        #     rtol,
        # )
        # print(h)
        # errors *= h[:, None]
        # print(errors.shape, h.shape)

        # if which_errors == "defect":
        #     errors, reference = estimate_errors_via_defect(
        #         bvp_posterior,
        #         kalman_posterior,
        #         candidate_locations,
        #         sigma_squared,
        #         measmod,
        #     )
        # elif which_errors == "probabilistic_defect":
        #     errors, reference = estimate_errors_via_probabilistic_defect(
        #         bvp_posterior, kalman_posterior, candidate_locations, sigma_squared, measmod
        #     )
        # else:
        #     errors, reference = estimate_errors_via_std(
        #         bvp_posterior,
        #         kalman_posterior,
        #         candidate_locations,
        #         sigma_squared,
        #         measmod,
        #     )

        # magnitude = stopcrit_bvp.evaluate_error(error=errors, reference=reference)
        # quotient = stopcrit_bvp.evaluate_quotient(errors, reference).squeeze()
        # # print(quotient)
        # quotient = np.linalg.norm(quotient, axis=1)

        mask = refinement_function(quotient)

        # kalman.initrv = randvars.Normal(
        #     mean=kalman_posterior[0].mean,
        #     cov=kalman.initrv.cov,
        #     cov_cholesky=kalman.initrv.cov_cholesky,
        # )

        yield bvp_posterior, sigma_squared, integral_error, kalman_posterior, candidate_locations, h, quotient, sigmas, insert_one, insert_two


ESTIMATE_ERRORS_OPTIONS = {
    "std": estimate_errors_via_std,
    "defect": estimate_errors_via_defect,
    "probabilistic_defect": estimate_errors_via_probabilistic_defect,
}
