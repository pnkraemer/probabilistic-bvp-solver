import numpy as np
from probnum import diffeq, randvars, utils, statespace
from probnum._randomvariablelist import _RandomVariableList

import functools

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
    quadrature,
)

import scipy.linalg


class BVPSolver:
    def __init__(
        self,
        dynamics_model,
        error_estimator,
        quadrature_rule,
        initialisation_strategy,
        initial_sigma_squared=1e10,
        use_bridge=True,
    ):
        self.dynamics_model = dynamics_model
        self.quadrature_rule = quadrature_rule
        self.initial_sigma_squared = initial_sigma_squared
        self.use_bridge = use_bridge

    @classmethod
    def from_default_values(
        cls,
        dynamics_model,
        initial_sigma_squared=1e10,
        use_bridge=True,
    ):

        return cls(
            dynamics_model=dynamics_model,
            error_estimator=error_estimates.estimate_errors_via_probabilistic_defect,
            quadrature_rule=quadrature.expquad_interior_only(),
            initialisation_strategy=bvp_initialise.bvp_initialise_ode,
            initial_sigma_squared=initial_sigma_squared,
            use_bridge=use_bridge,
        )

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    def solution_generator(
        self,
        bvp,
        atol,
        rtol,
        initial_grid,
        maxit_ieks=10,
        maxit_em=1,
        initial_guess=None,
    ):

        # Create data and measmods
        ode_measmod, left_measmod, right_measmod = self.choose_measurement_model(bvp)
        times = initial_grid
        dataset = np.zeros((len(times), bvp.dimension))
        measmod_list = self.create_measmod_list(
            ode_measmod, left_measmod, right_measmod, times
        )

        # Initialise with ODE or data
        filter_object, initrv = self.setup_filter_object(bvp)
        if initial_guess is None:
            kalman_posterior = filter_object.filtsmooth(
                dataset=dataset, times=times, measmod_list=measmod_list
            )
        else:
            initial_guess_measmod_list = initial_guess_measurement_models(
                initial_guess, self.dynamics_model, damping=0.0
            )
            kalman_posterior = filter_object.filtsmooth(
                dataset=initial_guess,
                times=times,
                measmod_list=initial_guess_measmod_list,
            )
        sigmas = filter_object.sigmas
        normalisation = filter_object.normalisation_for_sigmas
        sigma_squared = np.sum(sigmas) / normalisation
        yield kalman_posterior, sigma_squared

        while True:

            iter_em = 0
            iter_ieks = 0

            # EM iterations
            while iter_em < maxit_em:
                iter_em += 1

                # IEKS iterations
                while iter_ieks < maxit_ieks:
                    iter_ieks += 1
                    lin_measmod_list = self.linearise_measmod_list(
                        measmod_list, kalman_posterior.states, times
                    )
                    kalman_posterior = filter_object.filtsmooth(
                        dataset=dataset, times=times, measmod_list=lin_measmod_list
                    )

                initrv = self.update_initrv(kalman_posterior, initrv)
                filter_object.initrv = initrv

            yield kalman_posterior, sigma_squared

            # Recalibrate diffusion
            sigmas = filter_object.sigmas
            sigma_squared = np.mean(sigmas) / bvp.dimension
            initrv = self.update_covariances_with_sigma_squared(initrv, sigma_squared)

            # Compute errors
            _, mesh_candidates = insert_quadrature_nodes(
                mesh=times,
                quadrule=self.quadrule,
                where=np.ones_like(times[:-1], dtype=bool),
            )
            non_integrated_squared_error, reference, info = self.estimate_squared_error(
                kalman_posterior, mesh_candidates
            )
            normalised_squared_error = non_integrated_squared_error / (
                atol + rtol * reference
            )
            integrand = (
                np.linalg.norm(normalised_squared_error, axis=1) ** 2 / bvp.dimension
            )

            per_interval_error = (
                integrand.reshape((-1, self.quadrule.order - 2)) @ self.quadrule.weights
            )

            # Refine mesh wherever appropriate
            nu = self.dynamics_model.ordint
            threshold_two = 3.0 ** (nu + 0.5)
            acceptable = per_interval_error < 1.0
            insert_one = np.logical_and(
                1.0 < per_interval_error, per_interval_error < threshold_two
            )
            insert_two = threshold_two <= per_interval_error

            a1, _ = insert_quadrature_nodes(
                mesh=times, quadrule=self.quadrule[[1]], where=insert_one
            )
            a2, _ = insert_quadrature_nodes(
                mesh=times, quadrule=self.quadrule[[0], [2]], where=insert_two
            )
            times = functools.reduce(np.union1d, (times, a1, a2))
            dataset = np.zeros((len(times), bvp.dimension))
            measmod_list = self.create_measmod_list(
                ode_measmod, left_measmod, right_measmod, times
            )
            raise RuntimeError("Linearise at the previous evaluation!")

    #
    #
    #
    #
    #
    #
    #

    def setup_filter_object(self, bvp):
        initrv_not_bridged = self.create_initrv()
        initrv_not_bridged = self.update_covariances_with_sigma_squared(
            initrv_not_bridged, self.initial_sigma_squared
        )
        prior, initrv = self.initialise_bridge(bvp, initrv_not_bridged)
        filter_object = kalman.MyKalman(prior, None, initrv)
        return filter_object, initrv

    def create_initrv(self):
        m0 = np.ones(self.dynamics_model.dimension)
        c0 = self.initial_sigma_squared * np.ones(self.dynamics_model.dimension)
        C0 = np.diag(c0)
        initrv_not_bridged = randvars.Normal(m0, C0, cov_cholesky=np.sqrt(C0))
        return initrv_not_bridged

    def update_covariances_with_sigma_squared(self, initrv_not_bridged, sigma_squared):
        """Include sigma into initial covariance and process noise."""

        sigma = np.sqrt(sigma_squared)
        m0 = initrv_not_bridged.mean
        C0 = sigma_squared * initrv_not_bridged.cov
        C0_cholesky = sigma * initrv_not_bridged.cov_cholesky
        initrv_not_bridged = randvars.Normal(m0, C0, cov_cholesky=C0_cholesky)

        self.dynamics_model.equivalent_discretisation_preconditioned._proc_noise_cov_cholesky *= (
            sigma
        )
        self.dynamics_model.equivalent_discretisation_preconditioned.proc_noise_cov_mat *= (
            sigma_squared
        )
        return initrv_not_bridged

    def initialise_bridge(self, bvp, initrv_not_bridged):

        if self.use_bridge:
            bridge_prior = bridges.GaussMarkovBridge(self.dynamics_model, bvp)
            initrv_bridged = bridge_prior.initialise_boundary_conditions(
                initrv_not_bridged
            )
            return bridge_prior, initrv_bridged
        else:
            return self.dynamics_model, initrv_not_bridged

    def choose_measurement_model(self, bvp):

        if isinstance(bvp, problems.SecondOrderBoundaryValueProblem):
            ode_measmod = ode_measmods.from_second_order_ode(bvp, self.dynamics_model)
        else:
            ode_measmod = ode_measmods.from_ode(bvp, self.dynamics_model)

        left_measmod, right_measmod = ode_measmods.from_boundary_conditions(
            bvp, self.dynamics_model
        )
        return ode_measmod, left_measmod, right_measmod

    def create_measmod_list(self, ode_measmod, left_measmod, right_measmod, times):

        N = len(times)
        if N < 3:
            raise ValueError("Too few time steps")
        if self.use_bridge:
            return [ode_measmod] * N
        else:
            measmod_list = [left_measmod]
            measmod_list.extend([ode_measmod] * (N - 2))
            measmod_list.append(right_measmod)
            return measmod_list

    def linearise_measmod_list(self, measmod_list, states, times):

        if self.use_bridge:
            lin_measmod_list = [
                mm.linearize(state) for (mm, state) in zip(measmod_list, states)
            ]
        else:
            lin_measmod_list = [
                mm.linearize(state)
                for (mm, state) in zip(measmod_list[1:-1], states[1:-1])
            ]
            lin_measmod_list.insert(0, measmod_list[0])
            lin_measmod_list.append(measmod_list[-1])
        return lin_measmod_list

    def update_initrv(self, kalman_posterior, previous_initrv):
        """EM update for initial RV."""

        inferred_initrv = kalman_posterior.states[0]

        new_mean = inferred_initrv.mean
        new_cov_cholesky = utils.linalg.cholesky_update(
            inferred_initrv.cov_cholesky,
            inferred_initrv.mean - previous_initrv.mean,
        )
        new_cov = new_cov_cholesky @ new_cov_cholesky.T

        return randvars.Normal(
            mean=new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky
        )

    def estimate_squared_error(self, kalman_posterior, mesh_candidates, sigma_squared):
        evaluated_posterior = kalman_posterior(mesh_candidates)

        squared_error = evaluated_posterior.var * sigma_squared
        reference = evaluated_posterior.mean
        info = {"evaluated_posterior": evaluated_posterior}
        return squared_error, reference, info


def insert_quadrature_nodes(mesh, quadrule, where):
    """Insert 5-pt Lobatto points into a mesh."""

    print(quadrule)
    diff = np.diff(mesh)
    new_mesh = mesh

    for node in quadrule.nodes:
        new_pts = mesh[:-1] + diff * node
        new_mesh = np.union1d(new_mesh, new_pts[where])

    mesh_candidates = np.setdiff1d(new_mesh, mesh)
    return new_mesh, mesh_candidates


def initial_guess_measurement_models(initial_guess, prior, damping=0.0):
    N, d = initial_guess.shape
    projmat = prior.proj2coord(0)

    empty_shift = np.zeros(d)
    variances = damping * np.ones(d)
    process_noise_cov = np.diag(variances)
    process_noise_cov_cholesky = np.diag(np.sqrt(variances))

    single_measmod = statespace.DiscreteLTIGaussian(
        state_trans_mat=projmat,
        shift_vec=empty_shift,
        proc_noise_cov_mat=process_noise_cov,
        proc_noise_cov_cholesky=process_noise_cov_cholesky,
    )
    return [single_measmod] * N