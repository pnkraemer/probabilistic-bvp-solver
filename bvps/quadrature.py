"""Quadrature rules for error estimation."""


from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class QuadratureRule:

    nodes: np.ndarray
    weights: np.ndarray
    info: Dict = None


def gauss_lobatto_interior_only():
    LOBATTO_WEIGHTS = np.array([49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0])
    LOBATTO_NODES = np.array(
        [
            (np.sqrt(7.0) - np.sqrt(3)) / np.sqrt(28),
            1.0 / 2.0,
            (np.sqrt(7.0) + np.sqrt(3)) / np.sqrt(28),
        ]
    )
    return QuadratureRule(nodes=LOBATTO_NODES, weights=LOBATTO_WEIGHTS)