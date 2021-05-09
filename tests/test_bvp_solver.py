"""Test for BVP solver."""

from bvps import bvp_solver, problem_examples
import numpy as np

import pytest
from probnum import statespace, filtsmooth


@pytest.fixture
def solver():
    ibm3 = statespace.IBM(ordint=3, spatialdim=1)
    solver = bvp_solver.BVPSolver.from_default_values(ibm3, use_bridge=True)
    return solver


@pytest.fixture
def bvp():
    return problem_examples.problem_7_second_order(xi=0.1)


def test_bvp_solver(solver):
    assert isinstance(solver, bvp_solver.BVPSolver)


def test_choose_measurement_model(solver, bvp):
    ode, left, right = solver.choose_measurement_model(bvp)

    assert isinstance(ode, filtsmooth.DiscreteEKFComponent)
    assert isinstance(left, statespace.DiscreteLTIGaussian)
    assert isinstance(right, statespace.DiscreteLTIGaussian)

    dummy_times_array = np.arange(0.0, 100.0, 14)

    # Once with the bridge
    measmod_list = solver.create_measmod_list(ode, left, right, times=dummy_times_array)
    assert isinstance(measmod_list, list)
    assert len(measmod_list) == len(dummy_times_array)
    assert isinstance(measmod_list[0], filtsmooth.DiscreteEKFComponent)
    assert isinstance(measmod_list[1], filtsmooth.DiscreteEKFComponent)
    assert isinstance(measmod_list[-2], filtsmooth.DiscreteEKFComponent)
    assert isinstance(measmod_list[-1], filtsmooth.DiscreteEKFComponent)

    # And once without
    solver.use_bridge = False
    measmod_list = solver.create_measmod_list(ode, left, right, times=dummy_times_array)
    assert isinstance(measmod_list, list)
    assert len(measmod_list) == len(dummy_times_array)
    assert isinstance(measmod_list[0], statespace.DiscreteLTIGaussian)
    assert isinstance(measmod_list[1], filtsmooth.DiscreteEKFComponent)
    assert isinstance(measmod_list[-2], filtsmooth.DiscreteEKFComponent)
    assert isinstance(measmod_list[-1], statespace.DiscreteLTIGaussian)

    # For safety -- no idea how long the objects live
    solver.use_bridge = True