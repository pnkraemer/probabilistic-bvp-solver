import numpy as np
from probnum import diffeq, randvars, utils, statespace
from probnum._randomvariablelist import _RandomVariableList

import functools


import abc
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
        initial_sigma_squared=1e10,
        use_bridge=True,
    ):
        self.dynamics_model = dynamics_model
        self.error_estimator = error_estimator
        self.initial_sigma_squared = initial_sigma_squared
        self.use_bridge = use_bridge

        self.localconvrate = self.dynamics_model.ordint  # + 0.5?

    @classmethod
    def from_default_values_std_refinement(
        cls,
        dynamics_model,
        initial_sigma_squared=1e10,
        use_bridge=True,
        normalise_with_interval_size=True,
    ):
        quadrature_rule = quadrature.expquad_interior_only()
        P0 = dynamics_model.proj2coord(0)
        P1 = dynamics_model.proj2coord(1)
        error_estimator = ErrorViaStandardDeviation(
            atol=None,
            rtol=None,
            quadrature_rule=quadrature_rule,
            P0=P0,
            P1=P1,
            normalise_with_interval_size=normalise_with_interval_size,
        )
        return cls(
            dynamics_model=dynamics_model,
            error_estimator=error_estimator,
            initial_sigma_squared=initial_sigma_squared,
            use_bridge=use_bridge,
        )

    @classmethod
    def from_default_values(
        cls,
        dynamics_model,
        initial_sigma_squared=1e10,
        use_bridge=True,
        normalise_with_interval_size=True,
    ):
        quadrature_rule = quadrature.expquad_interior_only()
        P0 = dynamics_model.proj2coord(0)
        P1 = dynamics_model.proj2coord(1)
        error_estimator = ErrorViaResidual(
            atol=None,
            rtol=None,
            quadrature_rule=quadrature_rule,
            P0=P0,
            P1=P1,
            normalise_with_interval_size=normalise_with_interval_size,
        )
        return cls(
            dynamics_model=dynamics_model,
            error_estimator=error_estimator,
            initial_sigma_squared=initial_sigma_squared,
            use_bridge=use_bridge,
        )

    @classmethod
    def from_default_values_probabilistic_refinement(
        cls,
        dynamics_model,
        initial_sigma_squared=1e10,
        use_bridge=True,
        normalise_with_interval_size=True,
    ):
        quadrature_rule = quadrature.expquad_interior_only()
        P0 = dynamics_model.proj2coord(0)
        P1 = dynamics_model.proj2coord(1)
        error_estimator = ErrorViaProbabilisticResidual(
            atol=None,
            rtol=None,
            quadrature_rule=quadrature_rule,
            P0=P0,
            P1=P1,
            normalise_with_interval_size=normalise_with_interval_size,
        )
        return cls(
            dynamics_model=dynamics_model,
            error_estimator=error_estimator,
            initial_sigma_squared=initial_sigma_squared,
            use_bridge=use_bridge,
        )

    def solve(self, *args, **kwargs):
        for kalman_posterior, _ in self.solution_generator(*args, **kwargs):
            pass
        return kalman_posterior

    def solution_generator(
        self,
        bvp,
        atol,
        rtol,
        initial_grid,
        maxit_ieks=10,
        maxit_em=1,
        initial_guess=None,
        yield_ieks_iterations=False,
    ):

        self.error_estimator.set_tolerance(atol=atol, rtol=rtol)
        # Create data and measmods
        ode_measmod, left_measmod, right_measmod = self.choose_measurement_model(bvp)
        times = initial_grid
        dataset = np.zeros((len(times), bvp.dimension))
        measmod_list = self.create_measmod_list(
            ode_measmod, left_measmod, right_measmod, times
        )

        # Initialise with ODE or data
        filter_object = self.setup_filter_object(bvp)
        if initial_guess is None:
            kalman_posterior = filter_object.filtsmooth(
                dataset=dataset, times=times, measmod_list=measmod_list
            )
        else:
            initial_guess_measmod_list = self.initial_guess_measurement_models(
                initial_guess,
                damping=1e-14,
                left_measmod=left_measmod,
                right_measmod=right_measmod,
            )
            kalman_posterior = filter_object.filtsmooth(
                dataset=np.zeros_like(initial_guess),
                times=times,
                measmod_list=initial_guess_measmod_list,
            )
        sigmas = filter_object.sigmas
        normalisation = filter_object.normalisation_for_sigmas
        sigma_squared = np.sum(sigmas) / normalisation
        yield kalman_posterior, sigma_squared

        linearise_at = kalman_posterior.states

        acceptable_intervals = np.zeros(len(times[1:]), dtype=bool)
        while np.any(np.logical_not(acceptable_intervals)):
            # while True:
            # EM iterations
            for _ in range(maxit_em):

                # IEKS iterations
                for _ in range(maxit_ieks):

                    lin_measmod_list = self.linearise_measmod_list(
                        measmod_list, linearise_at, times
                    )
                    kalman_posterior = filter_object.filtsmooth(
                        dataset=dataset, times=times, measmod_list=lin_measmod_list
                    )
                    sigmas = filter_object.sigmas
                    sigma_squared = np.mean(sigmas) / bvp.dimension

                    linearise_at = kalman_posterior.states

                    if yield_ieks_iterations:
                        yield kalman_posterior, sigma_squared

                filter_object.initrv = self.update_initrv(
                    kalman_posterior, filter_object.initrv
                )

            yield kalman_posterior, sigma_squared

            # Recalibrate diffusion
            filter_object.initrv = self.update_covariances_with_sigma_squared(
                filter_object.initrv, sigma_squared
            )

            candidate_nodes = construct_candidate_nodes(
                current_mesh=times,
                nodes_per_interval=self.error_estimator.quadrature_rule.nodes,
            )
            evaluated_posterior = kalman_posterior(candidate_nodes)
            mm_list = [ode_measmod] * len(candidate_nodes)
            (
                per_interval_error,
                acceptable,
            ) = self.error_estimator.estimate_error_per_interval(
                evaluated_posterior,
                candidate_nodes,
                times,
                sigma_squared,
                ode_measmod_list=mm_list,
            )
            times, acceptable_intervals = refine_mesh(
                current_mesh=times,
                error_per_interval=per_interval_error,
                localconvrate=self.localconvrate,
                quadrature_nodes=self.error_estimator.quadrature_rule.nodes,
            )

            dataset = np.zeros((len(times), bvp.dimension))
            measmod_list = self.create_measmod_list(
                ode_measmod, left_measmod, right_measmod, times
            )
            linearise_at = kalman_posterior(times)

    def setup_filter_object(self, bvp):
        initrv_not_bridged = self.create_initrv()
        initrv_not_bridged = self.update_covariances_with_sigma_squared(
            initrv_not_bridged, self.initial_sigma_squared
        )
        prior, initrv = self.initialise_bridge(bvp, initrv_not_bridged)
        filter_object = kalman.MyKalman(prior, None, initrv)
        return filter_object

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
            measmod_list = [[left_measmod, ode_measmod]]
            measmod_list.extend([ode_measmod] * (N - 2))
            measmod_list.extend([[right_measmod, ode_measmod]])
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

            mm0 = measmod_list[0][0]
            lm0 = measmod_list[0][1].linearize(states[0])
            mm1 = measmod_list[-1][0]
            lm1 = measmod_list[-1][1].linearize(states[-1])

            lin_measmod_list.insert(0, [mm0, lm0])
            lin_measmod_list.append([mm1, lm1])

        return lin_measmod_list

    def update_initrv(self, kalman_posterior, previous_initrv):
        """EM update for initial RV."""

        inferred_initrv = kalman_posterior.states[0]

        new_mean = inferred_initrv.mean
        new_cov_cholesky = utils.linalg.cholesky_update(
            inferred_initrv.cov_cholesky,
            inferred_initrv.mean - previous_initrv.mean,
        )
        new_cov_cholesky += 1e-10 * np.eye(len(new_cov_cholesky))
        new_cov = new_cov_cholesky @ new_cov_cholesky.T

        return randvars.Normal(
            mean=new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky
        )

    #
    # def estimate_squared_error(self, kalman_posterior, mesh_candidates, sigma_squared):
    #     evaluated_posterior = kalman_posterior(mesh_candidates)
    #
    #     squared_error = evaluated_posterior.var * sigma_squared
    #     reference = evaluated_posterior.mean
    #     info = {"evaluated_posterior": evaluated_posterior}
    #     return squared_error, reference, info

    def initial_guess_measurement_models(
        self, initial_guess, damping, left_measmod, right_measmod
    ):

        N, d = initial_guess.shape
        projmat = self.dynamics_model.proj2coord(0)

        variances = damping * np.ones(d)
        process_noise_cov = np.diag(variances)
        process_noise_cov_cholesky = np.diag(np.sqrt(variances))

        measmodfun = lambda s: statespace.DiscreteLTIGaussian(
            state_trans_mat=projmat,
            shift_vec=s,
            proc_noise_cov_mat=process_noise_cov,
            proc_noise_cov_cholesky=process_noise_cov_cholesky,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
        if self.use_bridge:
            return [measmodfun(s=d) for d in initial_guess]

        else:
            measmod_list = [[left_measmod, measmodfun(s=initial_guess[0])]]
            measmod_list.extend([measmodfun(s=d) for d in initial_guess[1:-1]])
            measmod_list.extend([[right_measmod, measmodfun(s=initial_guess[-1])]])
            return measmod_list


########################################################################
########################################################################
# Mesh refinement
########################################################################
########################################################################

#
# def insert_quadrature_nodes(mesh, nodes_per_interval, where):
#     """Insert 5-pt Lobatto points into a mesh."""
#     new_candidates = construct_candidate_nodes(mesh, nodes_per_interval, where)
#     return np.union1d(mesh, new_candidates)
#


def refine_mesh(current_mesh, error_per_interval, localconvrate, quadrature_nodes):
    """Refine the mesh.

    Examples
    --------
    >>> current_mesh = [0., 0.5, 1.0, 2.0]
    >>> error_per_interval = [1000., 10., 0.1]
    >>> localconvrate = 3.5
    >>> quadrature_nodes = [0.3, 0.5, 0.7]
    >>> new_mesh, acceptable = refine_mesh(current_mesh, error_per_interval, localconvrate, quadrature_nodes)
    >>> print(new_mesh)
    [0.   0.15 0.35 0.5  0.75 1.   2.  ]
    >>> print(acceptable)
    [False False  True]
    """

    current_mesh = np.asarray(current_mesh)
    error_per_interval = np.asarray(error_per_interval)

    acceptable = error_per_interval < 1.0
    if np.all(acceptable):
        return current_mesh, acceptable

    threshold_two_instead_of_one = 3.0 ** localconvrate
    insert_one_here = np.logical_and(
        1.0 <= error_per_interval, error_per_interval <= threshold_two_instead_of_one
    )
    insert_two_here = threshold_two_instead_of_one < error_per_interval

    left_node, central_node, right_node = quadrature_nodes
    one_inserted = construct_candidate_nodes(
        current_mesh, [central_node], where=insert_one_here
    )
    two_inserted = construct_candidate_nodes(
        current_mesh, [left_node, right_node], where=insert_two_here
    )
    new_mesh = functools.reduce(np.union1d, (current_mesh, one_inserted, two_inserted))
    return new_mesh, acceptable


def construct_candidate_nodes(current_mesh, nodes_per_interval, where=None):
    """Construct nodes that are located in-between mesh points.

    Examples
    --------
    >>> current_mesh = [0., 0.5, 1.0, 2.0]
    >>> nodes_per_interval = [0.3, 0.5, 0.7]
    >>> candidate_nodes = construct_candidate_nodes(current_mesh, nodes_per_interval)
    >>> print(candidate_nodes)
    [0.15 0.25 0.35 0.65 0.75 0.85 1.3  1.5  1.7 ]

    >>> where = [True, False, False]
    >>> candidate_nodes = construct_candidate_nodes(current_mesh, nodes_per_interval, where=where)
    >>> print(candidate_nodes)
    [0.15 0.25 0.35]
    """

    current_mesh = np.asarray(current_mesh)
    if where is None:
        where = np.ones_like(current_mesh[1:], dtype=bool)

    diff = np.diff(current_mesh)
    new_mesh = []

    for node in nodes_per_interval:
        new_pts = current_mesh[:-1] + diff * node
        new_mesh = np.union1d(new_mesh, new_pts[where])
    return new_mesh


########################################################################
########################################################################
# Error estimation
########################################################################
########################################################################


class BVPErrorEstimator(abc.ABC):
    def __init__(
        self,
        atol,
        rtol,
        quadrature_rule,
        P0=None,
        P1=None,
        normalise_with_interval_size=True,
    ):
        self.quadrature_rule = quadrature_rule
        self.atol = atol
        self.rtol = rtol
        self.normalise_with_interval_size = normalise_with_interval_size

        # Projection matrices: state to 0th/1st derivative.
        self.P0 = P0
        self.P1 = P1

    def set_tolerance(self, atol, rtol):
        self.atol = atol
        self.rtol = rtol

    def estimate_error_per_interval(
        self,
        evaluated_posterior,
        mesh_candidates,
        current_mesh,
        calibrated_sigma_squared,
        ode_measmod_list=None,
    ):
        """Estimate error per interval.

        Numerically approximate the integrated error estimate per subinterval.
        """
        assert self.quadrature_rule.order == 5

        squared_error, reference, info = self.estimate_squared_error_at_points(
            evaluated_posterior,
            mesh_candidates,
            calibrated_sigma_squared,
            ode_measmod_list,
        )
        normalisation = (self.atol + self.rtol * np.abs(reference)) ** 2
        normalised_squared_error = squared_error / normalisation
        normalised_error = np.sqrt(normalised_squared_error)
        dim = len(normalised_error[0])

        integrand = np.linalg.norm(normalised_error, axis=1) ** 2 / dim
        per_interval_error = np.abs(
            integrand.reshape((-1, self.quadrature_rule.order - 2))
            @ self.quadrature_rule.weights
        )

        if self.normalise_with_interval_size:
            dt = np.diff(current_mesh)
            return np.sqrt(per_interval_error / dt), info

        return np.sqrt(per_interval_error), info

    @abc.abstractmethod
    def estimate_squared_error_at_points(
        self, evaluated_posterior, points, calibrated_sigma_squared
    ):
        raise NotImplementedError


class ErrorViaStandardDeviation(BVPErrorEstimator):
    """The posterior standard deviation is the error estimate.

    Examples
    --------
    >>> from probnum._randomvariablelist import _RandomVariableList
    >>> from probnum import randvars
    >>> from bvps import quadrature

    >>> quadrule = quadrature.QuadratureRule(nodes=[0.3, 0.5, 0.6], weights=[1./3., 1./3., 1./3.], order=5)
    >>> estimator = ErrorViaStandardDeviation(atol=0.5, rtol=0.5, quadrature_rule=quadrule)

    >>> dummy_rv = randvars.Normal(mean=np.ones(1), cov=np.eye(1))
    >>> evaluated_posterior = _RandomVariableList([dummy_rv]*12)
    >>> current_mesh = np.arange(0., 5., step=1.)
    >>> mesh_candidates = construct_candidate_nodes(current_mesh, quadrule.nodes)
    >>> calibrated_sigma_squared = 9
    >>> error, _ = estimator.estimate_error_per_interval(evaluated_posterior, mesh_candidates,current_mesh, calibrated_sigma_squared)
    >>> print(error)
    [3. 3. 3. 3.]
    >>> calibrated_sigma_squared = 100
    >>> error, _ = estimator.estimate_error_per_interval(evaluated_posterior, mesh_candidates,current_mesh, calibrated_sigma_squared)
    >>> print(error)
    [10. 10. 10. 10.]
    """

    def estimate_squared_error_at_points(
        self, evaluated_posterior, points, calibrated_sigma_squared, ode_measmod_list
    ):
        squared_error_estimate = (
            evaluated_posterior.var @ self.P0.T * calibrated_sigma_squared
        )
        reference = evaluated_posterior.mean @ self.P0.T
        return squared_error_estimate, reference, {}


class ErrorViaResidual(BVPErrorEstimator):
    """The size of the mean of the residual is the error estimate."""

    def estimate_squared_error_at_points(
        self, evaluated_posterior, points, calibrated_sigma_squared, ode_measmod_list
    ):
        assert len(ode_measmod_list) == len(evaluated_posterior)
        assert len(ode_measmod_list) == len(points)

        if self.P0 is None:
            raise ValueError("Pass a P0 to the ErrorEstimator.")
        residual_rv = _RandomVariableList(
            [
                mm.forward_rv(rv, t)[0]
                for mm, rv, t in zip(ode_measmod_list, evaluated_posterior, points)
            ]
        )
        squared_error_estimate = residual_rv.mean ** 2
        reference = evaluated_posterior.mean @ self.P0.T
        return squared_error_estimate, reference, {}


class ErrorViaProbabilisticResidual(BVPErrorEstimator):
    """The size of the mean of the residual is the error estimate."""

    def estimate_squared_error_at_points(
        self, evaluated_posterior, points, calibrated_sigma_squared, ode_measmod_list
    ):
        assert len(ode_measmod_list) == len(evaluated_posterior)
        assert len(ode_measmod_list) == len(points)

        if self.P0 is None:
            raise ValueError("Pass a P0 to the ErrorEstimator.")

        residual_rv = _RandomVariableList(
            [
                mm.forward_rv(rv, t)[0]
                for mm, rv, t in zip(ode_measmod_list, evaluated_posterior, points)
            ]
        )
        squared_error_estimate = (
            residual_rv.mean ** 2 + residual_rv.var * calibrated_sigma_squared
        )
        reference = evaluated_posterior.mean @ self.P0.T
        return squared_error_estimate, reference, {}
