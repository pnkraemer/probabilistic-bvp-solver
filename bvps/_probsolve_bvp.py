"""Solving BVPs."""
import numpy as np
from probnum import diffeq, randvars
from probnum._randomvariablelist import _RandomVariableList

from ._mesh import insert_single_points, insert_two_points, insert_three_points
from ._ode_measmods import from_ode, from_second_order_ode
from ._problems import SecondOrderBoundaryValueProblem
from ._probnum_overwrites import (
    ConstantStopping,
    MyIteratedDiscreteComponent,
    MyKalman,
    MyStoppingCriterion,
)

import scipy.linalg

def probsolve_bvp(
    bvp,
    bridge_prior,
    initial_grid,
    atol,
    rtol,
    maxit=50,
    which_method="iekf",
    insert="double",
    which_errors = "defect",
    ignore_bridge=False,
):
    """Solve a BVP.

    Parameters
    ----------
    bridge_prior : WrappedIntegrator
    which_method : either "ekf" or "iekf".
    insert : "single" or "double"
    """

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
        np.ones(bridge_prior.dimension), np.eye(bridge_prior.dimension)
    )
    initrv, _ = bridge_prior.forward_rv(rv, t=bvp.t0, dt=0.0)

    if ignore_bridge:
        print("No bridge detected.")
        kalman = MyKalman(
            dynamics_model=bridge_prior.integrator, measurement_model=measmod, initrv=rv
        )
    else:
        kalman = MyKalman(
            dynamics_model=bridge_prior, measurement_model=measmod, initrv=rv
        )


    # stopcrit_bvp = MyStoppingCriterion(atol=atol, rtol=rtol, maxit=maxit)
    stopcrit_bvp = ConstantStopping(maxit=maxit)

    # Call IEKS ##############################

    grid = initial_grid
    # stopcrit_ieks = MyStoppingCriterion(atol=100 * atol, rtol=100 * rtol, maxit=maxit)
    stopcrit_ieks = ConstantStopping(maxit=maxit)

    # Initial solve
    data = np.zeros((len(grid), bvp_dim)) 
    if ignore_bridge:
        kalman_posterior = kalman.iterated_filtsmooth(
            dataset=data, times=grid, stopcrit=stopcrit_ieks, measmodL=bridge_prior.measmod_L, measmodR=bridge_prior.measmod_R
        )
    else:
        kalman_posterior = kalman.iterated_filtsmooth(
            dataset=data, times=grid, stopcrit=stopcrit_ieks, measmodL=None, measmodR=None
        )

    bvp_posterior = diffeq.KalmanODESolution(kalman_posterior)
    sigma_squared = kalman.ssq

    # Set up candidates for mesh refinement
    if insert == "single":
        candidate_locations = insert_single_points(bvp_posterior.locations)
    elif insert == "double":
        candidate_locations = insert_two_points(bvp_posterior.locations)
    else:
        candidate_locations = insert_three_points(bvp_posterior.locations)

    # Estimate errors and choose nodes to refine
    if which_errors == "defect":
        errors, reference = estimate_errors_via_defect(bvp_posterior, kalman_posterior, candidate_locations, sigma_squared, measmod)
    else:
        errors, reference = estimate_errors_via_std(bvp_posterior, kalman_posterior, candidate_locations, sigma_squared, measmod)
        
    yield bvp_posterior, sigma_squared, errors, kalman_posterior, candidate_locations

    magnitude = stopcrit_bvp.evaluate_error(
        error=errors, reference=reference
    )
    quotient = stopcrit_bvp.evaluate_quotient(errors, reference)
    norm = np.linalg.norm(quotient, axis=1)
    mask = norm > np.median(norm)

    # while np.any(mask):
    while True:

        # Refine grid
        new_points = candidate_locations[mask]

        # new_fullgrid = np.union1d(grid, new_points)
        # sparse_fullgrid = np.union1d(new_fullgrid, new_fullgrid[:-1] + 0.5*np.diff(new_fullgrid))[::3]
        # grid = np.union1d(sparse_fullgrid, grid[[0, -1]])

        grid = np.union1d(grid, new_points)
        data = np.zeros((len(grid), bvp_dim))


        # Compute new solution
        if ignore_bridge:
            kalman_posterior = kalman.iterated_filtsmooth(
                dataset=data, times=grid, stopcrit=stopcrit_ieks, measmodL=bridge_prior.measmod_L, measmodR=bridge_prior.measmod_R
            )
        else:
            kalman_posterior = kalman.iterated_filtsmooth(
                dataset=data, times=grid, stopcrit=stopcrit_ieks, measmodL=None, measmodR=None
            )
        bvp_posterior = diffeq.KalmanODESolution(kalman_posterior)
        sigma_squared = kalman.ssq

        # Set up candidates for mesh refinement
        if insert == "single":
            candidate_locations = insert_single_points(bvp_posterior.locations)
        elif insert == "double":
            candidate_locations = insert_two_points(bvp_posterior.locations)
        else:
            candidate_locations = insert_three_points(bvp_posterior.locations)

        # Estimate errors and choose nodes to refine
        if which_errors == "defect":
            errors, reference = estimate_errors_via_defect(bvp_posterior, kalman_posterior, candidate_locations, sigma_squared, measmod)
        else:
            errors, reference = estimate_errors_via_std(bvp_posterior, kalman_posterior, candidate_locations, sigma_squared, measmod)



        magnitude = stopcrit_bvp.evaluate_error(
            error=errors, reference=reference
        )
        quotient = stopcrit_bvp.evaluate_quotient(errors, reference)

        norm = np.linalg.norm(quotient, axis=1)
        mask = norm > np.median(norm)


        yield bvp_posterior, sigma_squared, errors, kalman_posterior, candidate_locations



def estimate_errors_via_std(bvp_posterior, kalman_posterior, grid, ssq, measmod):
    evaluated_posterior = bvp_posterior(grid)
    errors = evaluated_posterior.std * np.sqrt(ssq)
    reference = evaluated_posterior.mean
    assert errors.shape == reference.shape
    return errors, evaluated_posterior.mean



def estimate_errors_via_defect(bvp_posterior, kalman_posterior, grid, ssq, measmod):
    evaluated_kalman_posterior = kalman_posterior(grid)
    msrvs = _RandomVariableList(
        [
            measmod.forward_realization(m, t=t)[0]
            for m, t in zip(evaluated_kalman_posterior.mean, grid)
        ]
    )
    errors = np.abs(msrvs.mean)
    reference = evaluated_kalman_posterior.mean @ kalman_posterior.transition.proj2coord(0).T
    assert errors.shape == reference.shape
    return errors, reference