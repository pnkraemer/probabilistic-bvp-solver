"""Solving BVPs."""
import numpy as np
from probnum import diffeq, randvars, utils
from probnum._randomvariablelist import _RandomVariableList

from ._mesh import insert_single_points, insert_two_points, insert_three_points
from ._ode_measmods import from_ode, from_second_order_ode
from ._problems import SecondOrderBoundaryValueProblem
from ._kalman import (
    ConstantStopping,
    MyIteratedDiscreteComponent,
    MyKalman,
    MyStoppingCriterion,
)

from ._bvp_initialise import *

import scipy.linalg


def refine_median(quotient):
    return quotient > np.median(quotient)


def refine_tolerance(quotient):
    return quotient > 1


REFINEMENT_OPTIONS = {"median": refine_median, "tolerance": refine_tolerance}


CANDIDATE_LOCATIONS_OPTIONS = {
    "single": insert_single_points,
    "double": insert_two_points,
    "triple": insert_three_points,
}


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
        10 * np.ones(bridge_prior.dimension), 1e6 * np.eye(bridge_prior.dimension)
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
    # stopcrit_ieks = MyStoppingCriterion(atol=100 * atol, rtol=100 * rtol, maxit=maxit)
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
    # kalman_posterior = bvp_posterior.kalman_posterior

    # Set up candidates for mesh refinement

    candidate_locations, h = candidate_function(bvp_posterior.locations)

    # Estimate errors and choose nodes to refine
    errors, reference = estimate_errors_function(
        bvp_posterior,
        kalman_posterior,
        candidate_locations,
        sigma_squared,
        measmod,
    )
    # errors *= h[:, None]

    print(errors.shape, h.shape)

    magnitude = stopcrit_bvp.evaluate_error(error=errors, reference=reference)
    quotient = stopcrit_bvp.evaluate_quotient(errors, reference)
    quotient = np.linalg.norm(quotient, axis=1)
    yield bvp_posterior, sigma_squared, errors, kalman_posterior, candidate_locations, h, quotient

    # print(quotient.shape)
    mask = refinement_function(quotient)
    print(mask)
    print(quotient)
    # print(quotient)
    # if refinement == "median":
    #     mask = refine_median(quotient)
    # else:
    #     mask = refine_tolerance(quotient)
    while np.any(mask):
        # while True:

        new_initrv = kalman_posterior.states[0]
        new_mean = new_initrv.mean.copy()
        # new_cov_cholesky = utils.linalg.cholesky_update(new_initrv.cov_cholesky, new_mean - kalman.initrv.mean)
        new_cov_cholesky = kalman.initrv.cov_cholesky
        new_cov = new_cov_cholesky @ new_cov_cholesky.T
        kalman.initrv = randvars.Normal(
            mean=new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky
        )

        # Refine grid
        print(mask.shape)
        new_points = candidate_locations[mask]

        # new_fullgrid = np.union1d(grid, new_points)
        # sparse_fullgrid = np.union1d(new_fullgrid, new_fullgrid[:-1] + 0.5*np.diff(new_fullgrid))[::3]
        # grid = np.union1d(sparse_fullgrid, grid[[0, -1]])

        grid = np.union1d(grid, new_points)
        print(grid.shape, new_points.shape, candidate_locations.shape)

        data = np.zeros((len(grid), bvp_dim))
        # Compute new solution
        if ignore_bridge:
            kalman_posterior = kalman.iterated_filtsmooth(
                dataset=data,
                times=grid,
                stopcrit=stopcrit_ieks,
                measmodL=bridge_prior.measmod_L,
                measmodR=bridge_prior.measmod_R,
            )
        else:
            kalman_posterior = kalman.iterated_filtsmooth(
                dataset=data,
                times=grid,
                stopcrit=stopcrit_ieks,
                measmodL=None,
                measmodR=None,
            )
        bvp_posterior = diffeq.KalmanODESolution(kalman_posterior)
        sigma_squared = kalman.ssq

        # Set up candidates for mesh refinement
        candidate_locations, h = candidate_function(bvp_posterior.locations)

        # Estimate errors and choose nodes to refine
        errors, reference = estimate_errors_function(
            bvp_posterior,
            kalman_posterior,
            candidate_locations,
            sigma_squared,
            measmod,
        )
        # print(h)
        # errors *= h[:, None]
        print(errors.shape, h.shape)

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

        magnitude = stopcrit_bvp.evaluate_error(error=errors, reference=reference)
        quotient = stopcrit_bvp.evaluate_quotient(errors, reference).squeeze()
        # print(quotient)
        quotient = np.linalg.norm(quotient, axis=1)

        mask = refinement_function(quotient)

        kalman.initrv = randvars.Normal(
            mean=kalman_posterior[0].mean,
            cov=kalman.initrv.cov,
            cov_cholesky=kalman.initrv.cov_cholesky,
        )

        yield bvp_posterior, sigma_squared, errors, kalman_posterior, candidate_locations, h, quotient


def estimate_errors_via_std(bvp_posterior, kalman_posterior, grid, ssq, measmod):
    evaluated_posterior = bvp_posterior(grid)
    errors = evaluated_posterior.std * np.sqrt(ssq)
    reference = evaluated_posterior.mean
    assert errors.shape == reference.shape
    return errors, evaluated_posterior.mean


def estimate_errors_via_defect(bvp_posterior, kalman_posterior, grid, ssq, measmod):
    h = np.amax(np.abs(np.diff(grid)))
    # print(h)
    evaluated_kalman_posterior = kalman_posterior(grid)
    msrvs = _RandomVariableList(
        [
            measmod.forward_realization(m, t=t)[0]
            for m, t in zip(evaluated_kalman_posterior.mean, grid)
        ]
    )
    errors = np.abs(msrvs.mean)
    reference = (
        evaluated_kalman_posterior.mean @ kalman_posterior.transition.proj2coord(0).T
    )
    assert errors.shape == reference.shape
    return errors, reference


def estimate_errors_via_probabilistic_defect(
    bvp_posterior, kalman_posterior, grid, ssq, measmod
):
    h = np.amax(np.abs(np.diff(grid)))
    evaluated_kalman_posterior = kalman_posterior(grid)
    msrvs = _RandomVariableList(
        [
            measmod.forward_rv(rv, t=t)[0]
            for rv, t in zip(evaluated_kalman_posterior, grid)
        ]
    )
    errors = np.sqrt(np.abs(msrvs.mean) ** 2 + np.abs(msrvs.std) ** 2)
    reference = (
        evaluated_kalman_posterior.mean @ kalman_posterior.transition.proj2coord(0).T
    )
    assert errors.shape == reference.shape
    return errors, reference


ESTIMATE_ERRORS_OPTIONS = {
    "std": estimate_errors_via_std,
    "defect": estimate_errors_via_defect,
    "probabilistic_defect": estimate_errors_via_probabilistic_defect,
}
