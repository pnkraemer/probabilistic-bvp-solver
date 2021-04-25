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
