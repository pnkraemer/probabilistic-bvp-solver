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
        initrv_not_bridged = self.initialise_sigma_squared()
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
        sigma_squared = np.inf
        yield kalman_posterior, sigma_squared

        while True:

            iter = 0
            while iter < maxit:
                iter += 1
                lin_measmod_list = self.linearise_measmod_list(
                    measmod_list, kalman_posterior.states, times
                )
                kalman_posterior = filter_object.filtsmooth(
                    dataset=dataset, times=times, measmod_list=lin_measmod_list
                )
                sigma_squared = np.inf
            yield kalman_posterior, sigma_squared

    #
    #
    #
    #
    #
    #
    #

    def initialise_sigma_squared(self):
        """Include sigma into initial covariance and process noise."""
        m0 = np.ones(self.dynamics_model.dimension)
        c0 = self.initial_sigma_squared * np.ones(self.dynamics_model.dimension)
        C0 = np.diag(c0)
        initrv_not_bridged = randvars.Normal(m0, C0, cov_cholesky=np.sqrt(C0))

        self.dynamics_model.equivalent_discretisation_preconditioned._proc_noise_cov_cholesky *= np.sqrt(
            self.initial_sigma_squared
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