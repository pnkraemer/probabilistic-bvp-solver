import dataclasses
from typing import Callable, Optional, Union

import numpy as np
import probnum.problems
from probnum.type import FloatArgType

from .problems import (
    BoundaryValueProblem,
    SecondOrderBoundaryValueProblem,
    FourthOrderBoundaryValueProblem,
)

# Check out: https://uk.mathworks.com/help/matlab/ref/bvp4c.html
# and: http://www.orcca.on.ca/TechReports/TechReports/2001/TR-01-02.pdf


####################################################################################################################


def pendulum():

    L = np.eye(1, 2)
    R = np.eye(1, 2)

    y0 = np.ones(1) * (-np.pi / 2)
    ymax = np.ones(1) * (np.pi / 2)
    t0 = 0.0
    tmax = np.pi / 2.0

    return BoundaryValueProblem(
        f=pendulum_rhs,
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df=pendulum_jac,
        dimension=2,
    )


def pendulum_rhs(t, y):
    return np.array([y[1], -9.81 * np.sin(y[0])])


def pendulum_jac(t, y):
    return np.array([[0.0, 1.0], [-9.81 * np.cos(y[0]), 0.0]])


def bratus(tmax=1.0):
    return bratus_second_order(tmax=tmax).to_first_order()


#     L = np.eye(1, 2)
#     R = np.eye(1, 2)
#     y0 = np.zeros(1)
#     ymax = np.zeros(1)
#     t0 = 0.0
#     tmax = tmax
#
#     return BoundaryValueProblem(
#         f=bratus_rhs,
#         t0=t0,
#         tmax=tmax,
#         L=L,
#         R=R,
#         y0=y0,
#         ymax=ymax,
#         df=bratus_jacobian,
#         dimension=2,
#     )
#
#
# def bratus_rhs(t, y):
#     return np.array([y[1], -np.exp(y[0])])
#
#
# def bratus_jacobian(t, y):
#     return np.array([[0.0, 1.0], [-np.exp(y[0]), 0.0]])


def bratus_second_order(tmax=1.0):

    L = np.eye(1, 2)
    R = np.eye(1, 2)
    y0 = np.zeros(1)
    ymax = np.zeros(1)
    t0 = 0.0
    tmax = tmax

    return SecondOrderBoundaryValueProblem(
        f=bratus_second_order_rhs,
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df_dy=bratus_second_order_jacobian_y,
        df_ddy=bratus_second_order_jacobian_dy,
        dimension=1,
    )


def bratus_second_order_rhs(t, y, dy):
    return -np.exp(y)


def bratus_second_order_jacobian_y(t, y, dy):
    return -np.exp(y) * np.ones((1, 1))


def bratus_second_order_jacobian_dy(t, y, dy):
    return 0.0 * np.ones((1, 1))


def matlab_example(tmax=1.0):
    """This has a closed form solution AND anisotropic behaviour (a lot happens in the beginning).
    Use this to show step-size adaptivity."""
    return matlab_example_second_order(tmax=tmax).to_first_order()


#     L = np.eye(1, 2)
#     R = np.eye(1, 2)
#
#     t0 = 1 / (3 * np.pi)
#     tmax = tmax
#
#     y0 = matlab_solution(t0)[0].reshape((-1,))
#     ymax = matlab_solution(tmax)[0].reshape((-1,))
#
#     return BoundaryValueProblem(
#         f=matlab_rhs,
#         t0=t0,
#         tmax=tmax,
#         L=L,
#         R=R,
#         y0=y0,
#         ymax=ymax,
#         df=matlab_jacobian,
#         solution=matlab_solution,
#         dimension=2,
#     )
#
#
# def matlab_rhs(t, y):
#     return np.array([y[1], -2 * y[1] / t - y[0] / t ** 4])
#
#
# def matlab_jacobian(t, y):
#     return np.array([[0, 1], [-1 / t ** 4, -2 / t]])
#
#
def matlab_solution(t):
    y1 = np.sin(1 / t)
    y2 = -1 / t ** 2 * np.cos(1 / t)
    return np.array([y1, y2])


def matlab_example_second_order(tmax=1.0):
    """This has a closed form solution AND anisotropic behaviour (a lot happens in the beginning).
    Use this to show step-size adaptivity."""

    L = np.eye(1, 2)
    R = np.eye(1, 2)

    t0 = 1 / (3 * np.pi)
    tmax = tmax

    y0 = matlab_solution(t0)[0].reshape((-1,))
    ymax = matlab_solution(tmax)[0].reshape((-1,))

    return SecondOrderBoundaryValueProblem(
        f=matlab_rhs_second_order,
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df_dy=matlab_jacobian_dy,
        df_ddy=matlab_jacobian_ddy,
        solution=matlab_solution_second_order,
        dimension=1,
    )


def matlab_rhs_second_order(t, y, dy):
    return -2 * dy / t - y / (t ** 4)


def matlab_jacobian_dy(t, y, dy):
    print(y.shape, dy.shape)
    return -1 / t ** 4 * np.ones((len(y), len(y)))


def matlab_jacobian_ddy(t, y, dy):
    return -2 / t * np.ones((len(y), len(y)))


def matlab_solution_second_order(t):
    return np.sin(1 / t)


def r_example(y0=None, ymax=None, xi=0.01):
    """https://rdrr.io/rforge/bvpSolve/f/inst/doc/bvpTests.pdf"""
    L = np.eye(1, 2)
    R = np.eye(1, 2)

    y0 = np.array([1.0]) if y0 is None else y0
    ymax = np.array([1.5]) if ymax is None else ymax

    t0 = 0.0
    tmax = 1.0

    return BoundaryValueProblem(
        f=lambda t, y: r_rhs(t, y, xi=xi),
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df=lambda t, y: r_jacobian(t, y, xi=xi),
        dimension=2,
    )


def r_rhs(t, y, xi):
    x, dx = y
    ynew = (x - x * dx) / xi
    return np.array([dx, ynew])


def r_jacobian(t, y, xi):
    x, dx = y
    ynew = (x - x * dx) / xi
    return np.array([[0, 1.0], [1 / xi - dx / xi, -x / xi]])


def problem_7(xi=0.1):
    """https://rdrr.io/rforge/bvpSolve/f/inst/doc/bvpTests.pdf"""
    return problem_7_second_order(xi=xi).to_first_order()


#     L = np.eye(1, 2)
#     R = np.eye(1, 2)
#
#     y0 = np.array([-1.0])
#     ymax = np.array([1.0])
#
#     t0 = -1.0
#     tmax = 1.0
#
#     return BoundaryValueProblem(
#         f=lambda t, y: p7_rhs(t, y, xi=xi),
#         t0=t0,
#         tmax=tmax,
#         L=L,
#         R=R,
#         y0=y0,
#         ymax=ymax,
#         df=lambda t, y: p7_jacobian(t, y, xi=xi),
#         dimension=2,
#     )
#
#
# def p7_rhs(t, y, xi):
#     x, dx = y
#     ynew = (
#         x
#         - t * dx
#         - (1 + xi * np.pi ** 2) * np.cos(np.pi * t)
#         - np.pi * t * np.sin(np.pi * t)
#     ) / xi
#     return np.array([dx, ynew])
#
#
# def p7_jacobian(t, y, xi):
#     x, dx = y
#     ynew = (x - x * dx) / xi
#     return np.array([[0, 1.0], [1 / xi, -t / xi]])


def problem_15(xi=0.01):
    """https://rdrr.io/rforge/bvpSolve/f/inst/doc/bvpTests.pdf"""
    L = np.eye(1, 2)
    R = np.eye(1, 2)

    y0 = np.array([1.0])
    ymax = np.array([1.0])

    t0 = -1.0
    tmax = 1.0

    return BoundaryValueProblem(
        f=lambda t, y: p15_rhs(t, y, xi=xi),
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df=lambda t, y: p15_jacobian(t, y, xi=xi),
        dimension=2,
    )


def p15_rhs(t, y, xi):
    x, dx = y
    ynew = (t * x) / xi
    return np.array([dx, ynew])


def p15_jacobian(t, y, xi):
    x, dx = y
    return np.array([[0, 1.0], [t / xi, 0.0]])


def problem_7_second_order(xi=0.1):
    """https://rdrr.io/rforge/bvpSolve/f/inst/doc/bvpTests.pdf"""
    L = np.eye(1, 2)
    R = np.eye(1, 2)

    y0 = np.array([-1.0])
    ymax = np.array([1.0])

    t0 = -1.0
    tmax = 1.0

    return SecondOrderBoundaryValueProblem(
        f=lambda t, y, dy: p7_rhs_second_order(t, y, dy, xi=xi),
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df_dy=lambda t, y, dy: p7_jacobian_second_order_dy(t, y, dy, xi=xi),
        df_ddy=lambda t, y, dy: p7_jacobian_second_order_ddy(t, y, dy, xi=xi),
        dimension=1,
    )


def p7_rhs_second_order(t, y, dy, xi):

    return (
        -(1 + xi * np.pi ** 2) * np.cos(np.pi * t)
        - np.pi * t * np.sin(np.pi * t)
        + y
        - t * dy
    ) / xi


def p7_jacobian_second_order_ddy(t, y, dy, xi):
    return np.ones((1, 1)) * -t / xi


def p7_jacobian_second_order_dy(t, y, dy, xi):
    return np.ones((1, 1)) / xi


###################################################


def seir(t0=0.0, tmax=55.0, y0=(97.0, 1.0, 1.0, 1.0), params=(0.3, 0.3, 0.1)):
    y0 = np.asarray(y0)
    population_count = np.sum(y0)
    params_and_population_count = (*params, population_count)

    def rhs(t, y):
        return seir_rhs(t, y, params_and_population_count)

    def jac(t, y):
        return seir_jac(t, y, params_and_population_count)

    return probnum.problems.InitialValueProblem(t0=t0, tmax=tmax, y0=y0, f=rhs, df=jac)


def seir_as_bvp(
    t0=0.0,
    tmax=55.0,
    IR0=(1.0, 1.0),
    IRmax=(10.0, 10.0),
    population_count=100,
    params=(0.3, 0.3, 0.1),
):

    # Projection to I & R
    L = np.flip(np.eye(2, 4))
    R = np.flip(np.eye(2, 4))

    y0 = np.asarray(IR0)
    ymax = np.asarray(IRmax)
    t0 = t0
    tmax = tmax

    params_and_population_count = (*params, population_count)

    def rhs(t, y):
        return seir_rhs(t, y, params_and_population_count)

    def jac(t, y):
        return seir_jac(t, y, params_and_population_count)

    return BoundaryValueProblem(
        f=rhs, t0=t0, tmax=tmax, L=L, R=R, y0=y0, ymax=ymax, df=jac, dimension=4
    )


def seir_rhs(t, y, params):
    """RHS for SEIR model."""
    alpha, beta, gamma, population_count = params
    y1, y2, y3, y4 = y
    y1_next = -beta * y1 * y3 / population_count
    y2_next = beta * y1 * y3 / population_count - alpha * y2
    y3_next = alpha * y2 - gamma * y3
    y4_next = gamma * y3

    return np.array([y1_next, y2_next, y3_next, y4_next])


def seir_jac(t, y, params):
    """Jacobian for SEIR model."""
    alpha, beta, gamma, population_count = params
    y1, y2, y3, y4 = y
    d_dy1 = np.array(
        [-beta * y3 / population_count, 0.0, -beta * y1 / population_count, 0.0]
    )
    d_dy2 = np.array(
        [beta * y3 / population_count, -alpha, beta * y1 / population_count, 0.0]
    )
    d_dy3 = np.array([0.0, alpha, -gamma, 0.0])
    d_dy4 = np.array([0.0, 0.0, gamma, 0.0])
    jac_matrix = np.array([d_dy1, d_dy2, d_dy3, d_dy4])
    return jac_matrix


def problem_20_second_order(xi=1.):
    """https://rdrr.io/rforge/bvpSolve/f/inst/doc/bvpTests.pdf"""
    L = np.eye(1, 2)
    R = np.eye(1, 2)

    y0 = np.array([1.0 + xi * np.log(np.cosh(-0.745 / xi))])
    ymax = np.array([1.0 + xi * np.log(np.cosh(0.255 / xi))])

    t0 = 0.0
    tmax = 1.0

    return SecondOrderBoundaryValueProblem(
        f=lambda t, y, dy: p20_rhs_second_order(t, y, dy, xi=xi),
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df_dy=lambda t, y, dy: p20_jacobian_second_order_dy(t, y, dy, xi=xi),
        df_ddy=lambda t, y, dy: p20_jacobian_second_order_ddy(t, y, dy, xi=xi),
        solution=lambda t: p20_solution(t, xi=xi),
        dimension=1,
    )


def p20_rhs_second_order(t, y, dy, xi):
    return (1 - dy ** 2) / xi


def p20_jacobian_second_order_ddy(t, y, dy, xi):
    return np.ones((1, 1)) * -2.0 * dy / xi


def p20_jacobian_second_order_dy(t, y, dy, xi):
    return np.ones((1, 1)) * 0.0


def p20_solution(t, xi):
    return 1 + xi * np.log(np.cosh((t - 0.745) / xi))


def problem_24_second_order(xi=0.5, gamma=1.4):
    """https://rdrr.io/rforge/bvpSolve/f/inst/doc/bvpTests.pdf"""
    L = np.eye(1, 2)
    R = np.eye(1, 2)

    y0 = np.array([0.9129])
    ymax = np.array([0.375])

    t0 = 0.0
    tmax = 1.0

    return SecondOrderBoundaryValueProblem(
        f=lambda t, y, dy: p24_rhs_second_order(t, y, dy, xi=xi, gamma=gamma),
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df_dy=lambda t, y, dy: p24_jacobian_second_order_dy(
            t, y, dy, xi=xi, gamma=gamma
        ),
        df_ddy=lambda t, y, dy: p24_jacobian_second_order_ddy(
            t, y, dy, xi=xi, gamma=gamma
        ),
        dimension=1,
    )


def p24_rhs_second_order(t, y, dy, xi, gamma):
    a = ((1 + gamma) / 2 - xi * dA(t)) * dy / (A(t) * xi)
    b = -dy / (y ** 2 * A(t) * xi)
    c = -dA(t) / (A(t) ** 2 * xi * y) + dA(t) / (A(t) ** 2 * xi) * (gamma - 1) / 2 * y
    return a + b + c


def p24_jacobian_second_order_ddy(t, y, dy, xi, gamma):
    da = ((1 + gamma) / 2 - xi * dA(t)) / (A(t) * xi)
    db = -1 / (y ** 2 * A(t) * xi)
    dc = 0
    return np.ones((1, 1)) * (da + db + dc)


def p24_jacobian_second_order_dy(t, y, dy, xi, gamma):
    da = 0.0
    db = 2 * dy / (A(t) * xi) / y ** 3
    dc = dA(t) / (A(t) ** 2 * xi * y ** 2) + dA(t) / (A(t) ** 2 * xi) * (gamma - 1) / 2
    return np.ones((1, 1)) * (da + db + dc)


def A(t):
    return 1 + t ** 2


def dA(t):
    return 2 * t


def measles():

    mu = 0.02
    xi = 0.0279
    eta = 0.01
    beta0 = 1575
    betafun = lambda t: beta0 * (1 + np.cos(2 * np.pi * t))

    # Projection to I & R
    eye = np.eye(3, 6)
    L = eye - np.flip(eye)
    R = eye - np.flip(eye)

    assert L.shape == (3, 6)
    assert R.shape == (3, 6)

    y0 = np.zeros(3)
    ymax = np.zeros(3)
    t0 = 0.0
    tmax = 1.0

    return BoundaryValueProblem(
        f=lambda t, y: measles_rhs(t, y, mu=mu, betafun=betafun, xi=xi, eta=eta),
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df=lambda t, y: measles_jac(t, y, mu=mu, betafun=betafun, xi=xi, eta=eta),
        dimension=6,
    )


def measles_rhs(t, y, mu, betafun, xi, eta):
    beta = betafun(t)
    f = y.copy()
    y1, y2, y3, c1, c2, c3 = y[0], y[1], y[2], y[3], y[4], y[5]
    # assert np.isscalar(y1), y1
    # assert np.isscalar(y2)
    # assert np.isscalar(y3)
    # assert np.isscalar(c1)
    # assert np.isscalar(c2)
    # assert np.isscalar(c3)
    f1 = mu - beta * y1 * y3
    f2 = beta * y1 * y3 - y2 / xi
    f3 = y2 / xi - y3 / eta
    f[0] = f1
    f[1] = f2
    f[2] = f3
    f[3] *= 0
    f[4] *= 0
    f[5] *= 0
    return f


def measles_jac(t, y, mu, betafun, xi, eta):
    beta = betafun(t)
    y1, y2, y3, c1, c2, c3 = y

    assert np.isscalar(y1)
    assert np.isscalar(y2)
    assert np.isscalar(y3)
    assert np.isscalar(c1)
    assert np.isscalar(c2)
    assert np.isscalar(c3)
    df1_dy1 = -beta * y3
    df1_dy2 = 0.0
    df1_dy3 = -beta * y1

    df2_dy1 = beta * y3
    df2_dy2 = -1 / xi
    df2_dy3 = beta * y1

    df3_dy1 = 0.0
    df3_dy2 = 1.0 / xi
    df3_dy3 = -1.0 / eta
    df_dy = np.array(
        [
            [df1_dy1, df1_dy2, df1_dy3, 0.0, 0.0, 0.0],
            [df2_dy1, df2_dy2, df2_dy3, 0.0, 0.0, 0.0],
            [df3_dy1, df3_dy2, df3_dy3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return df_dy


##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################


def problem_28_second_order(xi=0.4):

    L = np.eye(1, 2)
    R = np.eye(1, 2)

    y0 = np.array([1.0])
    ymax = np.array([1.5])

    t0 = 0.0
    tmax = 1.0

    return SecondOrderBoundaryValueProblem(
        f=lambda t, y, dy: p28_rhs_second_order(t, y, dy, xi=xi),
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df_dy=lambda t, y, dy: p28_jacobian_second_order_dy(t, y, dy, xi=xi),
        df_ddy=lambda t, y, dy: p28_jacobian_second_order_ddy(t, y, dy, xi=xi),
        dimension=1,
    )


def p28_rhs_second_order(t, y, dy, xi):
    return y * (1 - dy) / xi


def p28_jacobian_second_order_ddy(t, y, dy, xi):
    return np.ones((1, 1)) * (-y / xi)


def p28_jacobian_second_order_dy(t, y, dy, xi):
    return np.ones((1, 1)) * (1 - dy) / xi


def problem_32_fourth_order(xi=0.25):

    L_y = np.eye(1, 1)
    R_y = np.eye(1, 1)
    L_dy = np.eye(1, 1)
    R_dy = np.eye(1, 1)

    y0 = np.array([0.0])
    ymax = np.array([1.0])
    dy0 = np.array([0.0])
    dymax = np.array([0.0])

    t0 = 0.0
    tmax = 1.0

    return FourthOrderBoundaryValueProblem(
        f=lambda t, y, dy, ddy, dddy: p32_rhs_fourth_order(t, y, dy, ddy, dddy, xi=xi),
        t0=t0,
        tmax=tmax,
        L_y=L_y,
        R_y=R_y,
        L_dy=L_dy,
        R_dy=R_dy,
        y0=y0,
        ymax=ymax,
        dy0=dy0,
        dymax=dymax,
        df_dy=lambda t, y, dy, ddy, dddy: p32_jac_fourth_order_dy(
            t, y, dy, ddy, dddy, xi=xi
        ),
        df_ddy=lambda t, y, dy, ddy, dddy: p32_jac_fourth_order_ddy(
            t, y, dy, ddy, dddy, xi=xi
        ),
        df_dddy=lambda t, y, dy, ddy, dddy: p32_jac_fourth_order_dddy(
            t, y, dy, ddy, dddy, xi=xi
        ),
        df_ddddy=lambda t, y, dy, ddy, dddy: p32_jac_fourth_order_ddddy(
            t, y, dy, ddy, dddy, xi=xi
        ),
        dimension=1,
    )


def p32_rhs_fourth_order(t, y, dy, ddy, dddy, xi):
    return (dy * ddy - y * dddy) / xi


def p32_jac_fourth_order_dy(t, y, dy, ddy, dddy, xi):
    return np.ones((1, 1)) * -dddy / xi


def p32_jac_fourth_order_ddy(t, y, dy, ddy, dddy, xi):
    return np.ones((1, 1)) * ddy / xi


def p32_jac_fourth_order_dddy(t, y, dy, ddy, dddy, xi):
    return np.ones((1, 1)) * dy / xi


def p32_jac_fourth_order_ddddy(t, y, dy, ddy, dddy, xi):
    return np.ones((1, 1)) * -y / xi
