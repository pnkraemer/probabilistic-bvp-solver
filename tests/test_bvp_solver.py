"""Test for BVP solver."""

from bvps import bvp_solver, problem_examples


import pytest
from probnum import statespace, filtsmooth


@pytest.fixture
def solver():
    ibm3 = statespace.IBM(ordint=3, spatialdim=1)
    solver = bvp_solver.BVPSolver.from_default_values(ibm3)
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