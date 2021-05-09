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
        self.error_estimator = error_estimator
        self.quadrature_rule = quadrature_rule
        self.initialisation_strategy = initialisation_strategy
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
            quadrature_rule=quadrature.gauss_lobatto_interior_only(),
            initialisation_strategy=bvp_initialise.bvp_initialise_ode,
            initial_sigma_squared=initial_sigma_squared,
            use_bridge=use_bridge,
        )

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    def solution_generator(
        self, bvp, atol, rtol, initial_grid, maxit=10, initial_guess=None
    ):

        # Set up filter object
        initrv_not_bridged = self.create_initrv()
        initrv_not_bridged = self.update_covariances_with_sigma_squared(
            initrv_not_bridged, self.initial_sigma_squared
        )
        prior, initrv = self.initialise_bridge(bvp, initrv_not_bridged)
        filter_object = kalman.MyKalman(prior, None, initrv)

        # Create data and measmods
        dataset = np.zeros((len(initial_grid), bvp.dimension))
        times = initial_grid
        ode_measmod, left_measmod, right_measmod = self.choose_measurement_model(bvp)
        measmod_list = self.create_measmod_list(
            ode_measmod, left_measmod, right_measmod, times
        )

        # Initialise with ODE
        kalman_posterior = filter_object.filtsmooth(
            dataset=dataset, times=times, measmod_list=measmod_list
        )
        sigmas = filter_object.sigmas
        sigma_squared = np.mean(sigmas) / bvp.dimension
        yield kalman_posterior, sigma_squared

        while True:

            iter_em = 0
            iter_ieks = 0

            # EM iterations
            while iter_em < maxit:
                iter_em += 1

                # IEKS iterations
                while iter_ieks < maxit:
                    iter_ieks += 1
                    lin_measmod_list = self.linearise_measmod_list(
                        measmod_list, kalman_posterior.states, times
                    )
                    kalman_posterior = filter_object.filtsmooth(
                        dataset=dataset, times=times, measmod_list=lin_measmod_list
                    )
                    sigmas = filter_object.sigmas
                    sigma_squared = np.mean(sigmas) / bvp.dimension

                initrv = self.update_initrv(kalman_posterior, initrv)
                filter_object.initrv = initrv

            yield kalman_posterior, sigma_squared

            # Recalibrate diffusion
            initrv = self.update_covariances_with_sigma_squared(initrv, sigma_squared)

            # Compute errors

            # Refine mesh wherever appropriate

    #
    #
    #
    #
    #
    #
    #

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


def insert_quadrature_nodes(mesh, quadrule, where):
    """Insert 5-pt Lobatto points into a mesh."""

    print(quadrule)
    diff = np.diff(mesh)
    new_mesh = mesh

    for node in quadrule.nodes:
        new_pts = mesh[:-1] + diff * node
        new_mesh = np.union1d(new_mesh, new_pts[where])

    return new_mesh
    # left = (np.sqrt(7.0) - np.sqrt(3)) / np.sqrt(28)
    # middle = 1.0 / 2.0
    # right = (np.sqrt(7.0) + np.sqrt(3)) / np.sqrt(28)

    # mesh_left = mesh[:-1] + diff * left
    # mesh_middle = mesh[:-1] + diff * middle
    # mesh_right = mesh[:-1] + diff * right

    # mesh_left_and_middle = np.union1d(mesh_left[where], mesh_middle[where])
    # lobatto = np.union1d(mesh_left_and_middle, mesh_right[where])
    # new_mesh = np.union1d(mesh, lobatto)

    # return new_mesh, lobatto, np.repeat(diff, 3)
