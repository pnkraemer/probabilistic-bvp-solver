"""Updated ODE measurement mdoels."""


import numpy as np
import scipy.linalg
from probnum import filtsmooth, statespace
from probnum._randomvariablelist import _RandomVariableList

from .problems import SecondOrderBoundaryValueProblem, FourthOrderBoundaryValueProblem


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

    if isinstance(bvp, FourthOrderBoundaryValueProblem):
        return from_boundary_conditions_fourth_order(
            bvp, prior, damping_value=damping_value
        )
    if isinstance(bvp, SecondOrderBoundaryValueProblem):
        P0 = prior.proj2coord(0)
        P1 = prior.proj2coord(1)
        proj = np.vstack((P0, P1))
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


def from_fourth_order_ode(ode, prior, damping_value=0.0):

    spatialdim = prior.spatialdim
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)
    h2 = prior.proj2coord(coord=2)
    h3 = prior.proj2coord(coord=3)
    h4 = prior.proj2coord(coord=4)

    def dyna(t, x):
        return h4 @ x - ode.f(t, h0 @ x, h1 @ x, h2 @ x, h3 @ x)

    def diff(t):
        SQ = diff_cholesky(t)
        return SQ @ SQ.T

    def diff_cholesky(t):
        return np.sqrt(damping_value) * np.eye(spatialdim)

    def jacobian(t, x):

        df_dy = ode.df_dy(t, h0 @ x, h1 @ x, h2 @ x, h3 @ x)
        df_ddy = ode.df_ddy(t, h0 @ x, h1 @ x, h2 @ x, h3 @ x)
        df_dddy = ode.df_dddy(t, h0 @ x, h1 @ x, h2 @ x, h3 @ x)
        df_ddddy = ode.df_ddddy(t, h0 @ x, h1 @ x, h2 @ x, h3 @ x)
        return h2 - df_dy @ h0 - df_ddy @ h1 - df_dddy @ h2 - df_ddddy @ h3

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


def from_boundary_conditions_fourth_order(bvp, prior, damping_value=0.0):

    P0 = prior.proj2coord(0)
    P1 = prior.proj2coord(1)
    P2 = prior.proj2coord(2)
    P3 = prior.proj2coord(3)
    proj = np.vstack((P0, P1, P2, p3))

    use_y0 = bvp.y0 is not None
    use_dy0 = bvp.dy0 is not None
    use_ddy0 = bvp.ddy0 is not None
    use_dddy0 = bvp.dddy0 is not None
    use_ymax = bvp.ymax is not None
    use_dymax = bvp.dymax is not None
    use_ddymax = bvp.ddymax is not None
    use_dddymax = bvp.dddymax is not None

    zeros = np.zeros((bvp.dimension, bvp.dimension))
    Ls = (
        Li if use else zeros
        for Li, use in zip(
            [bvp.L_y, bvp.L_dy, bvp.L_ddy, bvp.L_dddy],
            [use_y0, use_dy0, use_ddy0, use_dddy0],
        )
    )
    Rs = (
        Ri if use else zeros
        for Ri, use in zip(
            [bvp.R_y, bvp.R_dy, bvp.R_ddy, bvp.R_dddy],
            [use_ymax, use_dymax, use_ddymax, use_dddymax],
        )
    )

    L = np.hstack(tuple(Ls))
    R = np.hstack(tuple(Rs))

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
