"""Updated ODE measurement mdoels."""


import numpy as np
import scipy.linalg
from probnum import filtsmooth, statespace
from probnum._randomvariablelist import _RandomVariableList


from .problems import SecondOrderBoundaryValueProblem


def from_ode(ode, prior, damping_value=0.0):

    spatialdim = prior.spatialdim
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)

    def dyna(t, x):
        return h1 @ x - ode.f(t, h0 @ x)

    def diff(t):
        SQ = diff_cholesky(t)
        return SQ @ SQ.T

    def diff_cholesky(t):
        return np.sqrt(damping_value) * np.eye(spatialdim)

    def jacobian(t, x):

        return h1 - ode.df(t, h0 @ x) @ h0

    discrete_model = statespace.DiscreteGaussian(
        input_dim=prior.dimension,
        output_dim=spatialdim,
        state_trans_fun=dyna,
        proc_noise_cov_mat_fun=diff,
        jacob_state_trans_fun=jacobian,
        proc_noise_cov_cholesky_fun=diff_cholesky,
    )
    return filtsmooth.DiscreteEKFComponent(
        discrete_model,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )


def from_second_order_ode(ode, prior, damping_value=0.0):

    spatialdim = prior.spatialdim
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)
    h2 = prior.proj2coord(coord=2)

    def dyna(t, x):
        return h2 @ x - ode.f(t, h0 @ x, h1 @ x)

    def diff(t):
        SQ = diff_cholesky(t)
        return SQ @ SQ.T

    def diff_cholesky(t):
        return np.sqrt(damping_value) * np.eye(spatialdim)

    def jacobian(t, x):
        return (
            h2 - ode.df_dy(t, h0 @ x, h1 @ x) @ h0 - ode.df_ddy(t, h0 @ x, h1 @ x) @ h1
        )

    discrete_model = statespace.DiscreteGaussian(
        input_dim=prior.dimension,
        output_dim=spatialdim,
        state_trans_fun=dyna,
        proc_noise_cov_mat_fun=diff,
        jacob_state_trans_fun=jacobian,
        proc_noise_cov_cholesky_fun=diff_cholesky,
    )
    return filtsmooth.DiscreteEKFComponent(
        discrete_model,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )


def from_boundary_conditions(bvp, prior, damping_value=0.0):

    if isinstance(bvp, SecondOrderBoundaryValueProblem):
        proj = np.stack((prior.proj2coord(0)[0], prior.proj2coord(1)[0]))
    else:
        proj = prior.proj2coord(0)

    L, R = bvp.L, bvp.R

    Rnew = R @ proj
    Lnew = L @ proj

    measmod_R = statespace.DiscreteLTIGaussian(
        Rnew,
        -bvp.ymax,
        damping_value * np.eye(len(R)),
        proc_noise_cov_cholesky=np.sqrt(damping_value) * np.eye(len(R)),
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
    measmod_L = statespace.DiscreteLTIGaussian(
        Lnew,
        -bvp.y0,
        damping_value * np.eye(len(L)),
        proc_noise_cov_cholesky=np.sqrt(damping_value) * np.eye(len(L)),
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
    return measmod_L, measmod_R
