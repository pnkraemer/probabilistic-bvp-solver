"""Assert that the jacobians are implemented correctly."""
import pytest

from bvps.problem_testset import testset_firstorder
import numpy as np


@pytest.fixture
def dt():
    return 1e-6


@pytest.fixture
def rtol():
    return 1e-6


@pytest.fixture
def bvp1st():
    return testset_firstorder()


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