import dataclasses
import numpy as np

from probnum.type import FloatArgType

from typing import Union, Optional, Callable


__all__ = ["BoundaryValueProblem", "bratus"]


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
    ddf: Optional[Callable[[float, np.ndarray], np.ndarray]] = None

    # For testing and benchmarking
    solution: Optional[Callable[[float, np.ndarray], np.ndarray]] = None

    @property
    def scipy_bc(self):

        def bc(ya, yb):

            return np.array([self.L @ ya - self.y0, self.R @ yb - self.ymax]).squeeze()
        return bc



def bratus(tmax=1.):

    L = np.eye(1, 2)
    R = np.eye(1, 2)
    y0 = np.zeros(1)
    ymax = np.zeros(1)
    t0 = 0.0
    tmax = tmax

    random_direction = np.random.rand(2)


    return BoundaryValueProblem(
        f=bratus_rhs, t0=t0, tmax=tmax, L=L, R=R, y0=y0, ymax=ymax, df=bratus_jacobian
    )


def bratus_rhs(t, y):
    return np.array([y[1], -np.exp(y[0])])

def bratus_jacobian(t, y):
    return np.array([[0., 1.], [-np.exp(y[0]), 0.]])
