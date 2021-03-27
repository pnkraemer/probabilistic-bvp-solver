import dataclasses
from typing import Callable, Optional, Union

import numpy as np
from probnum.type import FloatArgType

__all__ = [
    "BoundaryValueProblem",
    # "SecondOrderBoundaryValueProblem",
    "bratus",
    "matlab_example",
    "r_example",
    "problem_7",
]


# Check out: https://uk.mathworks.com/help/matlab/ref/bvp4c.html
# and: http://www.orcca.on.ca/TechReports/TechReports/2001/TR-01-02.pdf


@dataclasses.dataclass
class BoundaryValueProblem:
    """Boundary value problems."""

    f: Callable[[float, np.ndarray], np.ndarray]
    t0: float
    tmax: float

    L: np.ndarray  # projection to initial boundary value
    R: np.ndarray  # projection to final boundary value

    y0: Union[FloatArgType, np.ndarray]
    ymax: Union[FloatArgType, np.ndarray]
    df: Optional[Callable[[float, np.ndarray], np.ndarray]] = None

    # For testing and benchmarking
    solution: Optional[Callable[[float], np.ndarray]] = None

    @property
    def scipy_bc(self):
        def bc(ya, yb):

            return np.array([self.L @ ya - self.y0, self.R @ yb - self.ymax]).squeeze()

        return bc


@dataclasses.dataclass
class SecondOrderBoundaryValueProblem:
    """2nd order boundary value problems. Now, f=f(t, y, dy)."""

    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray]
    t0: float
    tmax: float

    L: np.ndarray  # projection from y to initial boundary value
    R: np.ndarray  # projection from y to final boundary value

    y0: Union[FloatArgType, np.ndarray]
    ymax: Union[FloatArgType, np.ndarray]
    df_dy: Optional[Callable[[float, np.ndarray, np.ndarray], np.ndarray]] = None
    df_ddy: Optional[Callable[[float, np.ndarray, np.ndarray], np.ndarray]] = None

    # For testing and benchmarking
    solution: Optional[Callable[[float], np.ndarray]] = None

    @property
    def scipy_bc(self):
        def bc(ya, yb):

            return np.array([self.L @ ya - self.y0, self.R @ yb - self.ymax]).squeeze()

        return bc


####################################################################################################################


def bratus(tmax=1.0):

    L = np.eye(1, 2)
    R = np.eye(1, 2)
    y0 = np.zeros(1)
    ymax = np.zeros(1)
    t0 = 0.0
    tmax = tmax

    return BoundaryValueProblem(
        f=bratus_rhs, t0=t0, tmax=tmax, L=L, R=R, y0=y0, ymax=ymax, df=bratus_jacobian
    )


def bratus_rhs(t, y):
    return np.array([y[1], -np.exp(y[0])])


def bratus_jacobian(t, y):
    return np.array([[0.0, 1.0], [-np.exp(y[0]), 0.0]])


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
    )


def bratus_second_order_rhs(t, y, dy):
    return -np.exp(y)


def bratus_second_order_jacobian_y(t, y, dy):
    return -np.exp(y) * np.eye(1)


def bratus_second_order_jacobian_dy(t, y, dy):
    return 0.0 * np.eye(1)


def matlab_example(tmax=1.0):
    """This has a closed form solution AND anisotropic behaviour (a lot happens in the beginning).
    Use this to show step-size adaptivity."""

    L = np.eye(1, 2)
    R = np.eye(1, 2)

    y0 = np.zeros(1)
    ymax = np.array([np.sin(1.0)])

    t0 = 1 / (3 * np.pi)
    tmax = tmax

    return BoundaryValueProblem(
        f=matlab_rhs,
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df=matlab_jacobian,
        solution=matlab_solution,
    )


def matlab_rhs(t, y):
    return np.array([y[1], -2 * y[1] / t - y[0] / t ** 4])


def matlab_jacobian(t, y):
    return np.array([[0, 1], [-1 / t ** 4, -2 / t]])


def matlab_solution(t):
    y1 = np.sin(1 / t)
    y2 = -1 / t ** 2 * np.cos(1 / t)
    return np.array([y1, y2])


def matlab_example_second_order(tmax=1.0):
    """This has a closed form solution AND anisotropic behaviour (a lot happens in the beginning).
    Use this to show step-size adaptivity."""

    L = np.eye(1, 2)
    R = np.eye(1, 2)

    y0 = np.zeros(1)
    ymax = np.array([np.sin(1.0)])

    t0 = 1 / (3 * np.pi)
    tmax = tmax

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
        solution=matlab_solution,
    )


def matlab_rhs_second_order(t, y, dy):
    return -2 * dy / t - y / (t ** 4)


def matlab_jacobian_dy(t, y, dy):
    return -1 / t ** 4 * np.ones((len(y), len(y)))


def matlab_jacobian_ddy(t, y, dy):
    return -2 / t * np.ones((len(y), len(y)))


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
    )


def r_rhs(t, y, xi):
    x, dx = y
    ynew = (x - x * dx) / xi
    return np.array([dx, ynew])


def r_jacobian(t, y, xi):
    x, dx = y
    ynew = (x - x * dx) / xi
    return np.array([[0, 1.0], [1 / xi - dx / xi, -x / xi]])


def problem_7(xi=0.01):
    """https://rdrr.io/rforge/bvpSolve/f/inst/doc/bvpTests.pdf"""
    L = np.eye(1, 2)
    R = np.eye(1, 2)

    y0 = np.array([-1.0])
    ymax = np.array([1.0])

    t0 = -1.0
    tmax = 1.0

    return BoundaryValueProblem(
        f=lambda t, y: p7_rhs(t, y, xi=xi),
        t0=t0,
        tmax=tmax,
        L=L,
        R=R,
        y0=y0,
        ymax=ymax,
        df=lambda t, y: p7_jacobian(t, y, xi=xi),
    )


def p7_rhs(t, y, xi):
    x, dx = y
    ynew = (
        x
        - t * dx
        - (1 + xi * np.pi ** 2) * np.cos(np.pi * t)
        - np.pi * t * np.sin(np.pi * t)
    ) / xi
    return np.array([dx, ynew])


def p7_jacobian(t, y, xi):
    x, dx = y
    ynew = (x - x * dx) / xi
    return np.array([[0, 1.0], [1 / xi, -t / xi]])


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
    )


def p15_rhs(t, y, xi):
    x, dx = y
    ynew = (t * x) / xi
    return np.array([dx, ynew])


def p15_jacobian(t, y, xi):
    x, dx = y
    return np.array([[0, 1.0], [t / xi, 0.0]])


def problem_7_second_order(xi=0.01):
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
    )


def p7_rhs_second_order(t, y, dy, xi):

    return (
        -(1 + xi * np.pi ** 2) * np.cos(np.pi * t)
        - np.pi * t * np.sin(np.pi * t)
        + y
        - t * dy
    ) / xi


def p7_jacobian_second_order_dy(t, y, dy, xi):
    return np.ones((1, 1)) * -t


def p7_jacobian_second_order_ddy(t, y, dy, xi):
    return np.ones((1, 1))
