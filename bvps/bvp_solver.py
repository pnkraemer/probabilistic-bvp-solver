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
    def from_default_values(cls, dynamics_model):

        return cls(
            dynamics_model=dynamics_model,
            error_estimator=error_estimates.estimate_errors_via_probabilistic_defect,
            quadrature_rule=quadrature.gauss_lobatto_interior_only(),
            initialisation_strategy=bvp_initialise.bvp_initialise_ode,
        )

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    def solution_generator(
        self, bvp, atol, rtol, maxit=10, initial_grid=None, initial_guess=None
    ):

        initrv_not_bridged = self.initial_sigma_squared()
        prior, initrv = self.initialise_bridge(bvp, initrv_not_bridged)
        measmod = self.choose_measurement_model(bvp)
        kalman = kalman.MyKalman(prior, measmod, initrv)

        kalman_posterior, sigma_squared = self.initialisation_strategy(
            bvp=bvp,
            prior=prior,
            initial_grid=initial_grid,
            initrv=initrv,
            use_bridge=self.use_bridge,
        )
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
        if self.use_bridge:
            return [ode_measmod] * N

        else:
            return [left_measmod].extend([ode_measmod] * (N - 2)).append(right_measmod)
