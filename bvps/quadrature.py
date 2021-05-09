"""Quadrature rules for error estimation."""


from dataclasses import dataclass
from typing import Dict
import numpy as np
from probnum import kernels, quad


@dataclass
class QuadratureRule:

    nodes: np.ndarray
    weights: np.ndarray
    order: int
    info: Dict = None

    def __getitem__(self, idx):

        return QuadratureRule(
            nodes=nodes[idx], weights=weights[idx], order=None, info=info
        )


def gauss_lobatto_interior_only():
    LOBATTO_WEIGHTS = np.array([49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0])
    LOBATTO_NODES = np.array(
        [
            (np.sqrt(7.0) - np.sqrt(3)) / np.sqrt(28),
            1.0 / 2.0,
            (np.sqrt(7.0) + np.sqrt(3)) / np.sqrt(28),
        ]
    )
    return QuadratureRule(
        nodes=LOBATTO_NODES,
        weights=LOBATTO_WEIGHTS,
        order=5,
    )


def expquad_interior_only(expquad_lengthscale=1.0):

    # Construct objects
    gaussian_kernel = kernels.ExpQuad(input_dim=1, lengthscale=expquad_lengthscale)
    lebesgue_measure = quad.LebesgueMeasure(domain=[0.0, 1.0])
    kernel_embedding = quad.KernelEmbedding(gaussian_kernel, lebesgue_measure)

    # Choose grid and compute kernel embeddings
    grid = np.array([0.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 1.0]).reshape((-1, 1))
    mean_embedding = kernel_embedding.kernel_mean(grid)
    variance_embedding = kernel_embedding.kernel_variance()

    # Compute weights and (uncalibrated) posterior variance
    K = gaussian_kernel(grid, grid)
    Kinv = np.linalg.inv(K)
    weights = mean_embedding @ Kinv
    variance = np.abs(variance_embedding - weights @ mean_embedding)
    return QuadratureRule(
        nodes=grid[1:-1, 0],
        weights=weights[1:-1],
        info={"variance": variance},
        order=5,
    )
