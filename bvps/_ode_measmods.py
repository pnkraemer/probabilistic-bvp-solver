"""Updated ODE measurement mdoels."""


import numpy as np
import scipy.linalg
from probnum import filtsmooth, statespace
from probnum._randomvariablelist import _RandomVariableList


def from_ode(ode, prior):

    spatialdim = prior.spatialdim
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)

    def dyna(t, x):
        return h1 @ x - ode.f(t, h0 @ x)

    def diff_cholesky(t):
        return 0 * np.eye(spatialdim)

    def jacobian(t, x):

        return h1 - ode.df(t, h0 @ x) @ h0

    discrete_model = statespace.DiscreteGaussian(
        input_dim=prior.dimension,
        output_dim=spatialdim,
        state_trans_fun=dyna,
        proc_noise_cov_mat_fun=diff_cholesky,
        jacob_state_trans_fun=jacobian,
        proc_noise_cov_cholesky_fun=diff_cholesky,
    )
    return filtsmooth.DiscreteEKFComponent(
        discrete_model,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )


def from_second_order_ode(ode, prior):

    spatialdim = prior.spatialdim
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)
    h2 = prior.proj2coord(coord=2)

    def dyna(t, x):
        return h2 @ x - ode.f(t, h0 @ x, h1 @ x)

    def diff_cholesky(t):
        return 0 * np.eye(spatialdim)

    def jacobian(t, x):
        return (
            h2 - ode.df_dy(t, h0 @ x, h1 @ x) @ h0 - ode.df_ddy(t, h0 @ x, h1 @ x) @ h1
        )

    discrete_model = statespace.DiscreteGaussian(
        input_dim=prior.dimension,
        output_dim=spatialdim,
        state_trans_fun=dyna,
        proc_noise_cov_mat_fun=diff_cholesky,
        jacob_state_trans_fun=jacobian,
        proc_noise_cov_cholesky_fun=diff_cholesky,
    )
    return filtsmooth.DiscreteEKFComponent(
        discrete_model,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
