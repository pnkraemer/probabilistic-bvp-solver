"""Test for BVP solver."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from probnum import filtsmooth, randvars, statespace

from bvps import bvp_solver, problem_examples, quadrature


@pytest.fixture
def solver(use_bridge):
    ibm3 = statespace.IBM(
        ordint=4,
        spatialdim=1,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
    solver = bvp_solver.BVPSolver.from_default_values(ibm3, use_bridge=use_bridge)
    return solver


@pytest.fixture
def bvp():
    return problem_examples.problem_7_second_order(xi=1e-2)


@pytest.mark.parametrize("use_bridge", [True, False])
def test_bvp_solver(solver):
    assert isinstance(solver, bvp_solver.BVPSolver)


@pytest.mark.parametrize("use_bridge", [True, False])
def test_choose_measurement_model(solver, bvp):
    ode, left, right = solver.choose_measurement_model(bvp)

    assert isinstance(ode, filtsmooth.DiscreteEKFComponent)
    assert isinstance(left, statespace.DiscreteLTIGaussian)
    assert isinstance(right, statespace.DiscreteLTIGaussian)

    dummy_times_array = np.arange(0.0, 100.0, 14)
    measmod_list = solver.create_measmod_list(ode, left, right, times=dummy_times_array)

    if solver.use_bridge:
        assert isinstance(measmod_list, list)
        assert len(measmod_list) == len(dummy_times_array)
        assert isinstance(measmod_list[0], filtsmooth.DiscreteEKFComponent)
        assert isinstance(measmod_list[1], filtsmooth.DiscreteEKFComponent)
        assert isinstance(measmod_list[-2], filtsmooth.DiscreteEKFComponent)
        assert isinstance(measmod_list[-1], filtsmooth.DiscreteEKFComponent)

    if not solver.use_bridge:
        assert isinstance(measmod_list, list)
        assert len(measmod_list) == len(dummy_times_array)
        assert isinstance(measmod_list[0][0], statespace.DiscreteLTIGaussian)
        assert isinstance(measmod_list[0][1], filtsmooth.DiscreteEKFComponent)
        assert isinstance(measmod_list[-1][0], statespace.DiscreteLTIGaussian)
        assert isinstance(measmod_list[-1][1], filtsmooth.DiscreteEKFComponent)


@pytest.mark.parametrize("use_bridge", [True, False])
def test_linearize_measmod_list(bvp, solver):
    ode, left, right = solver.choose_measurement_model(bvp)
    N = 14
    dummy_times_array = np.arange(0.0, 100.0, N)
    measmod_list = solver.create_measmod_list(ode, left, right, times=dummy_times_array)

    dummy_states = [randvars.Constant(np.ones(solver.dynamics_model.dimension))] * N
    lin_measmod_list = solver.linearise_measmod_list(
        measmod_list, dummy_states, dummy_times_array
    )
    if solver.use_bridge:
        assert isinstance(lin_measmod_list, list)
        assert len(lin_measmod_list) == len(dummy_times_array)
        assert isinstance(lin_measmod_list[0], statespace.DiscreteLinearGaussian)
        assert isinstance(lin_measmod_list[1], statespace.DiscreteLinearGaussian)
        assert isinstance(lin_measmod_list[-2], statespace.DiscreteLinearGaussian)
        assert isinstance(lin_measmod_list[-1], statespace.DiscreteLinearGaussian)

    if not solver.use_bridge:
        assert isinstance(lin_measmod_list, list)
        assert len(lin_measmod_list) == len(dummy_times_array)
        assert isinstance(measmod_list[0][0], statespace.DiscreteLTIGaussian)
        assert isinstance(measmod_list[0][1], filtsmooth.DiscreteEKFComponent)
        assert isinstance(measmod_list[-1][0], statespace.DiscreteLTIGaussian)
        assert isinstance(measmod_list[-1][1], filtsmooth.DiscreteEKFComponent)


@pytest.mark.parametrize("use_bridge", [True, False])
def test_initialise(bvp, solver):

    N = 4
    dummy_initial_grid = np.linspace(bvp.t0, bvp.tmax, N)

    gen = solver.solution_generator(
        bvp, atol=1.0, rtol=1.0, initial_grid=dummy_initial_grid
    )
    kalman_posterior, sigma_squared = next(gen)

    t = kalman_posterior.locations
    y = kalman_posterior.states.mean
    # plt.title(f"Use bridge: {solver.use_bridge}, N:{len(t)}")
    # plt.plot(t, y[:, 0])
    # plt.plot(t, y[:, 1])
    # plt.show()
    assert t.shape == (N,)
    assert y.shape == (N, solver.dynamics_model.dimension)


@pytest.mark.parametrize("use_bridge", [True, False])
def test_first_iteration(bvp, solver):

    N = 5
    dummy_initial_grid = np.linspace(bvp.t0, bvp.tmax, N)

    # maxit_ieks is NaN, because this part of the code should never be reached,
    # if the initialisation is yielded properly.
    gen = solver.solution_generator(
        bvp,
        atol=1.0,
        rtol=1.0,
        initial_grid=dummy_initial_grid,
        maxit_ieks=np.nan,
    )

    kalman_posterior, sigma_squared = next(gen)
    t = kalman_posterior.locations
    y = kalman_posterior.states.mean

    # plt.title(f"Use bridge: {solver.use_bridge}, N:{len(t)}")
    # plt.plot(t, y[:, :3])
    # plt.plot(kalman_posterior.locations, kalman_posterior.states.mean[:, :3], "o", color="gray")
    # plt.show()

    assert t.shape == (N,)
    assert y.shape == (N, solver.dynamics_model.dimension)


@pytest.mark.parametrize("use_bridge", [True, False])
def test_full_iteration(bvp, solver):

    N = 5
    dummy_initial_grid = np.linspace(bvp.t0, bvp.tmax, N)

    gen = solver.solution_generator(
        bvp,
        atol=1.0,
        rtol=1.0,
        initial_grid=dummy_initial_grid,
        maxit_ieks=3,
    )

    for idx, (kalman_posterior, sigma_squared) in enumerate(gen):
        pass

    t = kalman_posterior.locations
    y = kalman_posterior.states.mean
    # plt.title(f"Use bridge: {solver.use_bridge}, NIter: {idx + 1}, N:{len(t)}")
    # plt.plot(t, y)
    # plt.show()

    N, d = len(t), solver.dynamics_model.dimension
    assert y.shape == (N, d)
