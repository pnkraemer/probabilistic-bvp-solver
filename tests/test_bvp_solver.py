"""Test for BVP solver."""

from bvps import bvp_solver


import pytest
from probnum import statespace


def test_bvp_solver():

    ibm3 = statespace.IBM(ordint=3, spatialdim=1)
    solver = bvp_solver.BVPSolver.from_default_values(ibm3)
    assert isinstance(solver, bvp_solver.BVPSolver)
