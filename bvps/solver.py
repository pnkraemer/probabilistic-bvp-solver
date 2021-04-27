"""Solving BVPs."""
import numpy as np
from probnum import diffeq, randvars, utils
from probnum._randomvariablelist import _RandomVariableList

from bvps import (
    mesh,
    ode_measmods,
    problems,
    kalman,
    bvp_initialise,
    bridges,
    control,
    error_estimates,
    stopcrit,
)

import scipy.linalg


def refine_median(quotient):
    return quotient > np.median(quotient)


def refine_tolerance(quotient):
    return quotient > 1


REFINEMENT_OPTIONS = {"median": refine_median, "tolerance": refine_tolerance}


CANDIDATE_LOCATIONS_OPTIONS = {
    "single": mesh.insert_central_point,
    "double": mesh.insert_two_equispaced_points,
    "triple": None,
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
    initial_sigma_squared=1e5,
):
    """Solve a BVP.

    Parameters
    ----------
    bridge_prior : WrappedIntegrator
    which_method : either "ekf" or "iekf".
    insert : "single" or "double"
    """
    # Set up a bridge prior with the correct sigma scaling
    m0 = np.zeros(bridge_prior.dimension)
    c0 = initial_sigma_squared * np.ones(bridge_prior.dimension)
    C0 = np.diag(c0)
    initrv_not_bridged = randvars.Normal(m0, C0, cov_cholesky=np.sqrt(C0))
    ibm = bridge_prior.integrator
    ibm.equivalent_discretisation_preconditioned._proc_noise_cov_cholesky *= np.sqrt(
        initial_sigma_squared
    )
    bridge_prior = bridges.GaussMarkovBridge(ibm, bvp)
    initrv = bridge_prior.initialise_boundary_conditions(initrv_not_bridged)

    # Choose a measurement model
    if isinstance(bvp, problems.SecondOrderBoundaryValueProblem):
        print("Recognised a 2nd order BVP")
        measmod = ode_measmods.from_second_order_ode(bvp, bridge_prior)

        bvp_dim = len(bvp.R.T) // 2
    else:
        measmod = ode_measmods.from_ode(bvp, bridge_prior)
        bvp_dim = len(bvp.R.T)
    if which_method == "iekf":
        stopcrit_iekf = stopcrit.ConstantStopping(maxit=maxit)
        measmod = kalman.MyIteratedDiscreteComponent(measmod, stopcrit=stopcrit_iekf)

    # Construct Kalman object ##############################
    if ignore_bridge:
        print("No bridge detected.")
        kalman_filtsmooth = kalman.MyKalman(
            dynamics_model=bridge_prior.integrator, measurement_model=measmod, initrv=rv
        )
    else:
        kalman_filtsmooth = kalman.MyKalman(
            dynamics_model=bridge_prior, measurement_model=measmod, initrv=initrv
        )

    # Choose control-related functions
    refinement_function = REFINEMENT_OPTIONS[refinement]
    estimate_errors_function = ESTIMATE_ERRORS_OPTIONS[which_errors]
    candidate_function = CANDIDATE_LOCATIONS_OPTIONS[insert]

    stopcrit_bvp = stopcrit.MyStoppingCriterion(atol=atol, rtol=rtol, maxit=maxit)
    # stopcrit_bvp = ConstantStopping(maxit=maxit)

    # Call IEKS ##############################

    grid = initial_grid
    # stopcrit_ieks = MyStoppingCriterion(atol=atol, rtol=rtol, maxit=maxit)
    stopcrit_ieks = stopcrit.ConstantStopping(maxit=maxit)

    # Initialise
    kalman_posterior = bvp_initialise.bvp_initialise_ode(
        bvp=bvp, bridge_prior=bridge_prior, initial_grid=initial_grid, initrv=initrv
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
    ) = control.control(
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
    yield bvp_posterior, sigma_squared, integral_error, kalman_posterior, candidate_locations, h, quotient, sigmas, insert_one, insert_two, measmod

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

        bridge_prior = bridges.GaussMarkovBridge(ibm, bvp)

        new_initrv = kalman_posterior.states[0]
        new_mean = new_initrv.mean.copy()

        # The damping value is added, because we initialise the bridge again right after (and Dirac-Dirac does not work)
        new_cov_cholesky = (
            utils.linalg.cholesky_update(
                sigma * new_initrv.cov_cholesky,
                new_mean - kalman_filtsmooth.initrv.mean,
            )
            + np.eye(len(new_mean)) * 1e-6
        )
        # new_cov_cholesky = kalman.initrv.cov_cholesky
        new_cov = new_cov_cholesky @ new_cov_cholesky.T
        initrv_not_initialised = randvars.Normal(
            mean=new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky
        )
        initrv = bridge_prior.initialise_boundary_conditions(initrv_not_initialised)
        kalman.initrv = initrv

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
            kalman_posterior = kalman_filtsmooth.iterated_filtsmooth(
                dataset=data,
                times=new_mesh,
                stopcrit=stopcrit_ieks,
                measmodL=bridge_prior.measmod_L,
                measmodR=bridge_prior.measmod_R,
                old_posterior=kalman_posterior,
            )
        else:
            kalman_posterior = kalman_filtsmooth.iterated_filtsmooth(
                dataset=data,
                times=new_mesh,
                stopcrit=stopcrit_ieks,
                measmodL=None,
                measmodR=None,
                old_posterior=kalman_posterior,
            )
        bvp_posterior = diffeq.KalmanODESolution(kalman_posterior)
        sigma_squared = kalman_filtsmooth.ssq
        sigmas = kalman_filtsmooth.sigmas

        (
            new_mesh,
            integral_error,
            quotient,
            candidate_locations,
            h,
            insert_one,
            insert_two,
        ) = control.control(
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

        yield bvp_posterior, sigma_squared, integral_error, kalman_posterior, candidate_locations, h, quotient, sigmas, insert_one, insert_two, measmod


ESTIMATE_ERRORS_OPTIONS = {
    "std": error_estimates.estimate_errors_via_std,
    "defect": error_estimates.estimate_errors_via_defect,
    "probabilistic_defect": error_estimates.estimate_errors_via_probabilistic_defect,
}
