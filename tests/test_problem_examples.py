"""Assert that the jacobians are implemented correctly."""
import pytest

from bvps.problem_examples import *
import numpy as np

all_first_order_bvps = pytest.mark.parametrize(
    "bvp1st",
    [
        pendulum(),
        bratus(),
        matlab_example(),
        r_example(),
        problem_7(),
        problem_15(),
        seir_as_bvp(),
        measles(),
    ],
)


all_second_order_bvps = pytest.mark.parametrize(
    "bvp2nd",
    [
        bratus_second_order(),
        matlab_example_second_order(),
        problem_7_second_order(),
        problem_20_second_order(),
        problem_24_second_order(),
    ],
)


@pytest.fixture
def dt():
    return 1e-6


@pytest.fixture
def rtol():
    return 1e-6


@all_first_order_bvps
def test_jacobians_1st(bvp1st, dt, rtol):

    bvp_dim = len(bvp1st.R.T)
    random_direction = 1 + 0.1 * np.random.rand(bvp_dim)
    random_point = 1 + np.random.rand(bvp_dim)

    f1 = bvp1st.f(bvp1st.t0, random_point + dt * random_direction)
    f2 = bvp1st.f(bvp1st.t0, random_point - dt * random_direction)
    fd_approx = (f1 - f2) / (2 * dt)

    true_df = bvp1st.df(bvp1st.t0, random_point)

    assert f1.ndim == 1
    assert f2.ndim == 1

    assert true_df.ndim == 2
    np.testing.assert_allclose(
        true_df @ random_direction,
        fd_approx,
        rtol=rtol,
    )


@all_second_order_bvps
def test_jacobians_2nd_dy(bvp2nd, dt, rtol):

    bvp_dim = len(bvp2nd.R.T) // 2
    random_direction = 1 + 0.1 * np.random.rand(bvp_dim)
    random_point = 1 + np.random.rand(bvp_dim)

    f1 = bvp2nd.f(bvp2nd.t0, random_point + dt * random_direction, random_point)
    f2 = bvp2nd.f(bvp2nd.t0, random_point - dt * random_direction, random_point)
    fd_approx = (f1 - f2) / (2 * dt)

    true_df = bvp2nd.df_dy(bvp2nd.t0, random_point, random_point)

    assert f1.ndim == 1
    assert f2.ndim == 1

    assert true_df.ndim == 2

    np.testing.assert_allclose(
        true_df @ random_direction,
        fd_approx,
        rtol=rtol,
    )


@all_second_order_bvps
def test_jacobians_2nd_ddy(bvp2nd, dt, rtol):

    bvp_dim = len(bvp2nd.R.T) // 2
    random_direction = 1 + 0.1 * np.random.rand(bvp_dim)
    random_point = 1 + np.random.rand(bvp_dim)

    f1 = bvp2nd.f(bvp2nd.t0, random_point, random_point + dt * random_direction)
    f2 = bvp2nd.f(bvp2nd.t0, random_point, random_point - dt * random_direction)
    fd_approx = (f1 - f2) / (2 * dt)

    true_df = bvp2nd.df_ddy(bvp2nd.t0, random_point, random_point)

    assert f1.ndim == 1
    assert f2.ndim == 1
    assert true_df.ndim == 2

    np.testing.assert_allclose(
        true_df @ random_direction,
        fd_approx,
        rtol=rtol,
    )
