"""Solving BVPs."""
from ._probnum_overwrites import (
    from_ode,
    MyStoppingCriterion,
    MyIteratedDiscreteComponent,
    MyKalman,
)
from probnum import randvars, diffeq
import numpy as np
from ._mesh import insert_single_points, insert_two_points


def probsolve_bvp(
    bvp,
    bridge_prior,
    initial_grid,
    atol,
    rtol,
    maxit=500,
    which_method="ekf",
    insert="single",
):
    """Solve a BVP.

    Parameters
    ----------
    bridge_prior : WrappedIntegrator
    which_method : either "ekf" or "iekf".
    insert : "single" or "double"
    """

    # Construct Kalman object ##############################

    measmod = from_ode(bvp, bridge_prior)
    if which_method == "iekf":
        stopcrit_iekf = MyStoppingCriterion(atol=atol, rtol=rtol, maxit=maxit)
        measmod = MyIteratedDiscreteComponent(measmod, stopcrit=stopcrit_iekf)

    rv = randvars.Normal(
        np.ones(bridge_prior.dimension), np.eye(bridge_prior.dimension)
    )
    initrv, _ = bridge_prior.forward_rv(rv, t=bvp.t0, dt=0.0)

    kalman = MyKalman(
        dynamics_model=bridge_prior, measurement_model=measmod, initrv=initrv
    )

    stopcrit_bvp = MyStoppingCriterion(atol=atol, rtol=rtol, maxit=maxit)

    # Call IEKS ##############################

    grid = initial_grid
    stopcrit_ieks = MyStoppingCriterion(atol=atol, rtol=rtol, maxit=maxit)
    bvp_dim = len(bvp.R.T)
    data = np.zeros((len(grid), bvp_dim))
    kalman_posterior = kalman.iterated_filtsmooth(
        dataset=data, times=grid, stopcrit=stopcrit_ieks
    )
    bvp_posterior = diffeq.KalmanODESolution(kalman_posterior)
    sigma_squared = kalman.ssq

    # Set up candidates for mesh refinement
    if insert == "single":
        candidate_locations = insert_single_points(bvp_posterior.locations)
    else:
        candidate_locations = insert_two_points(bvp_posterior.locations)
    evaluated_posterior = bvp_posterior(candidate_locations)
    errors = evaluated_posterior.std * np.sqrt(sigma_squared)
    magnitude = stopcrit_bvp.evaluate_error(
        error=errors, reference=evaluated_posterior.mean
    )
    quotient = stopcrit_bvp.evaluate_quotient(errors, evaluated_posterior.mean)
    mask = np.linalg.norm(quotient, axis=1) > np.sqrt(bvp_dim)

    while np.any(mask):
        new_points = candidate_locations[mask]
        grid = np.sort(np.append(grid, new_points))
        data = np.zeros((len(grid), bvp_dim))
        kalman_posterior = kalman.iterated_filtsmooth(
            dataset=data, times=grid, stopcrit=stopcrit_ieks
        )
        bvp_posterior = diffeq.KalmanODESolution(kalman_posterior)
        sigma_squared = kalman.ssq

        # Set up candidates for mesh refinement
        if insert == "single":
            candidate_locations = insert_single_points(bvp_posterior.locations)
        else:
            candidate_locations = insert_two_points(bvp_posterior.locations)
        evaluated_posterior = bvp_posterior(candidate_locations)
        errors = evaluated_posterior.std * np.sqrt(sigma_squared)
        magnitude = stopcrit_bvp.evaluate_error(
            error=errors, reference=evaluated_posterior.mean
        )
        quotient = stopcrit_bvp.evaluate_quotient(errors, evaluated_posterior.mean)
        mask = np.linalg.norm(quotient, axis=1) > np.sqrt(bvp_dim)
        # print(np.linalg.norm(quotient, axis=1))
        print(len(grid))

        # print(mask, len(grid))
    return bvp_posterior