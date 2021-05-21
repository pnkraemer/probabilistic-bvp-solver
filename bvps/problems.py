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

    @property
    def scipy_bc(self):
        def bc(ya, yb):

            return np.array([self.L @ ya - self.y0, self.R @ yb - self.ymax]).squeeze()

        return bc

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
        return np.block([[dx], [dy]])

    def _jac_as_firstorder(self, t, y):
        x, dx = y
        x = np.atleast_1d(x)
        dx = np.atleast_1d(dx)
        df_dy = self.df_dy(t, y=x, dy=dx)
        df_ddy = self.df_ddy(t, y=x, dy=dx)
        I = np.eye(self.dimension)
        O = np.zeros_like(I)
        return np.block([[O, I], [df_dy, df_ddy]])
