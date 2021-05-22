"""BVP Problem data types."""

import dataclasses
from typing import Callable, Optional, Union

import numpy as np
from probnum.type import FloatArgType


@dataclasses.dataclass
class BoundaryValueProblem:
    """Boundary value problems."""

    f: Callable[[float, np.ndarray], np.ndarray]
    t0: float
    tmax: float

    L: np.ndarray  # projection to initial boundary value
    R: np.ndarray  # projection to final boundary value
    dimension: int

    y0: Union[FloatArgType, np.ndarray]
    ymax: Union[FloatArgType, np.ndarray]
    df: Optional[Callable[[float, np.ndarray], np.ndarray]] = None

    # For testing and benchmarking
    solution: Optional[Callable[[float], np.ndarray]] = None

    @property
    def scipy_bc(self):
        def bc(ya, yb):
            X = np.array([self.L @ ya - self.y0, self.R @ yb - self.ymax]).flatten()
            return X

        return bc


@dataclasses.dataclass
class SecondOrderBoundaryValueProblem:
    """2nd order boundary value problems. Now, f=f(t, y, dy)."""

    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray]
    t0: float
    tmax: float

    L: np.ndarray  # projection from y to initial boundary value
    R: np.ndarray  # projection from y to final boundary value

    dimension: int

    y0: Union[FloatArgType, np.ndarray]
    ymax: Union[FloatArgType, np.ndarray]
    df_dy: Optional[Callable[[float, np.ndarray, np.ndarray], np.ndarray]] = None
    df_ddy: Optional[Callable[[float, np.ndarray, np.ndarray], np.ndarray]] = None

    # For testing and benchmarking
    solution: Optional[Callable[[float], np.ndarray]] = None

    def to_first_order(self):

        f = self._rhs_as_firstorder
        if self.df_dy is not None and self.df_ddy is not None:
            df = self._jac_as_firstorder
        else:
            df = None
        return BoundaryValueProblem(
            f=f,
            t0=self.t0,
            tmax=self.tmax,
            L=self.L,
            R=self.R,
            y0=self.y0,
            ymax=self.ymax,
            df=df,
            dimension=self.dimension * 2,
            solution=self.solution,
        )

    def _rhs_as_firstorder(self, t, y):
        x, dx = y
        x = np.atleast_1d(x)
        dx = np.atleast_1d(dx)
        dy = self.f(t=t, y=x, dy=dx)
        return np.block([[dx], [dy]]).squeeze()

    def _jac_as_firstorder(self, t, y):
        x, dx = y
        x = np.atleast_1d(x)
        dx = np.atleast_1d(dx)
        df_dy = self.df_dy(t, y=x, dy=dx)
        df_ddy = self.df_ddy(t, y=x, dy=dx)
        I = np.eye(self.dimension)
        O = np.zeros_like(I)
        return np.block([[O, I], [df_dy, df_ddy]])


@dataclasses.dataclass
class FourthOrderBoundaryValueProblem:
    """4th order boundary value problems. Now, f=f(t, y, dy, ddy, dddy)."""

    f: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    t0: float
    tmax: float
    dimension: int

    L_y: Optional[np.ndarray] = None
    R_y: Optional[np.ndarray] = None
    L_dy: Optional[np.ndarray] = None
    R_dy: Optional[np.ndarray] = None
    L_ddy: Optional[np.ndarray] = None
    R_ddy: Optional[np.ndarray] = None
    L_dddy: Optional[np.ndarray] = None
    R_dddy: Optional[np.ndarray] = None

    y0: Optional[np.ndarray] = None
    ymax: Optional[np.ndarray] = None
    dy0: Optional[np.ndarray] = None
    dymax: Optional[np.ndarray] = None
    ddy0: Optional[np.ndarray] = None
    ddymax: Optional[np.ndarray] = None
    dddy0: Optional[np.ndarray] = None
    dddymax: Optional[np.ndarray] = None

    df_dy: Optional[
        Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    ] = None
    df_ddy: Optional[
        Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    ] = None
    df_dddy: Optional[
        Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    ] = None
    df_ddddy: Optional[
        Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    ] = None

    # For testing and benchmarking
    solution: Optional[Callable[[float], np.ndarray]] = None

    def to_first_order(self):

        f = self._rhs_as_firstorder
        if self.df_dy is not None and self.df_ddy is not None:
            df = self._jac_as_firstorder
        else:
            df = None

        assert self.L_y is not None and self.L_dy is not None
        assert self.R_y is not None and self.R_dy is not None
        assert self.L_ddy is None and self.L_dddy is None
        assert self.R_ddy is None and self.R_dddy is None

        L = np.hstack((self.L_y, self.L_dy))
        R = np.hstack((self.R_y, self.R_dy))
        y0 = np.vstack((self.y0, self.dy0))
        ymax = np.hstack((self.ymax, self.dymax))
        return BoundaryValueProblem(
            f=f,
            t0=self.t0,
            tmax=self.tmax,
            L=L,
            R=R,
            y0=y0,
            ymax=ymax,
            df=df,
            dimension=self.dimension * 4,
            solution=self.solution,
        )

    def _rhs_as_firstorder(self, t, y):
        x, dx, ddx, dddx = y
        x = np.atleast_1d(x)
        dx = np.atleast_1d(dx)
        ddx = np.atleast_1d(ddx)
        dddx = np.atleast_1d(dddx)
        dy = self.f(t=t, y=x, dy=dx, ddy=ddx, dddy=dddx)
        return np.block([[dx], [ddx], [dddx], [dy]]).squeeze()

    def _jac_as_firstorder(self, t, y):
        x, dx, ddx, dddx = y
        x = np.atleast_1d(x)
        dx = np.atleast_1d(dx)
        ddx = np.atleast_1d(ddx)
        dddx = np.atleast_1d(dddx)

        df_dy = self.df_dy(t=t, y=x, dy=dx, ddy=ddx, dddy=dddx)
        df_ddy = self.df_ddy(t=t, y=x, dy=dx, ddy=ddx, dddy=dddx)
        df_dddy = self.df_dddy(t=t, y=x, dy=dx, ddy=ddx, dddy=dddx)
        df_ddddy = self.df_ddddy(t=t, y=x, dy=dx, ddy=ddx, dddy=dddx)

        I = np.eye(self.dimension)
        O = np.zeros_like(I)
        return np.block(
            [
                [O, I, O, O],
                [O, O, I, O],
                [O, O, O, I],
                [df_dy, df_ddy, df_dddy, df_ddddy],
            ]
        )
