"""Solving BVPs."""
import numpy as np
from probnum import diffeq, randvars
from probnum._randomvariablelist import _RandomVariableList

from ._mesh import insert_single_points, insert_two_points
from ._ode_measmods import from_ode, from_second_order_ode
from ._problems import SecondOrderBoundaryValueProblem
from ._probnum_overwrites import (
    ConstantStopping,
    MyIteratedDiscreteComponent,
    MyKalman,
    MyStoppingCriterion,
)


def probsolve_bvp(
    bvp,
    bridge_prior,
    initial_grid,
    atol,
    rtol,
    maxit=50,
    which_method="iekf",
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

    kalman = MyKalman(
        dynamics_model=bridge_prior, measurement_model=measmod, initrv=initrv
    )

    # stopcrit_bvp = MyStoppingCriterion(atol=atol, rtol=rtol, maxit=maxit)
    stopcrit_bvp = ConstantStopping(maxit=maxit)

    # Call IEKS ##############################

    grid = initial_grid
    # stopcrit_ieks = MyStoppingCriterion(atol=100 * atol, rtol=100 * rtol, maxit=maxit)
    stopcrit_ieks = ConstantStopping(maxit=maxit)

    data = np.zeros((len(grid), bvp_dim))

    print(len(grid))

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

    evaluated_kalman_posterior = kalman_posterior(candidate_locations)
    msrvs = _RandomVariableList(
        [
            measmod.forward_realization(m, t=t)[0]
            for m, t in zip(evaluated_kalman_posterior.mean, candidate_locations)
        ]
    )
    errors = np.abs(msrvs.mean * np.sqrt(sigma_squared))

    yield bvp_posterior, sigma_squared, errors

    magnitude = stopcrit_bvp.evaluate_error(
        error=errors, reference=evaluated_posterior.mean
    )
    quotient = stopcrit_bvp.evaluate_quotient(errors, evaluated_posterior.mean)
    mask = np.linalg.norm(quotient, axis=1) > np.sqrt(bvp_dim)

    # while np.any(mask):
    while True:
        new_points = candidate_locations[mask]
        grid = np.sort(np.append(grid, new_points))
        data = np.zeros((len(grid), bvp_dim))
        print(len(grid))

        # print(data.shape)
        # print(kalman.initrv.mean)
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

        evaluated_kalman_posterior = kalman_posterior(candidate_locations)
        msrvs = _RandomVariableList(
            [
                measmod.forward_realization(m, t=t)[0]
                for m, t in zip(evaluated_kalman_posterior.mean, candidate_locations)
            ]
        )
        errors = np.abs(msrvs.mean * np.sqrt(sigma_squared))

        magnitude = stopcrit_bvp.evaluate_error(
            error=errors, reference=evaluated_posterior.mean
        )
        quotient = stopcrit_bvp.evaluate_quotient(errors, evaluated_posterior.mean)
        # mask = np.linalg.norm(quotient, axis=1) > np.sqrt(bvp_dim)
        mask = np.linalg.norm(errors, axis=1) > np.median(
            np.linalg.norm(errors, axis=1)
        )
        # print(np.linalg.norm(quotient, axis=1))
        yield bvp_posterior, sigma_squared, errors
        # print(mask, len(grid))
    # return bvp_posterior
