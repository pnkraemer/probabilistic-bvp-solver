"""Mesh refinement and error control."""
import numpy as np

from .error_estimates import *
from .mesh import *

LOBATTO_WEIGHTS = np.array([49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0])


def control(bvp_posterior, kalman_posterior, ssq, measmod, atol, rtol):

    bvp_dim = measmod.output_dim
    _, new_candidates, dt = insert_lobatto5_points(
        bvp_posterior.locations,
        where=np.ones_like(bvp_posterior.locations[:-1], dtype=bool),
    )
    _, _, quotient = estimate_errors_via_probabilistic_defect(
        bvp_posterior, kalman_posterior, new_candidates, ssq, measmod, atol, rtol
    )
    per_interval_quotient = quotient.reshape((-1, 3))
    integral_error = np.sqrt(per_interval_quotient @ LOBATTO_WEIGHTS / (bvp_dim))

    nu = kalman_posterior.transition.ordint
    threshold = 3.0 ** (nu + 0.5)
    # print(threshold)
    # print(integral_error.shape)
    acceptable = integral_error < 1
    insert_one = np.logical_and(1 <= integral_error, integral_error < threshold)
    insert_two = threshold <= integral_error
    # print(insert_one.shape)

    # insert_one_points = np.union1d(
    #     bvp_posterior.locations[:-1][insert_one], bvp_posterior.locations[-1]
    # )
    # insert_two_points = np.union1d(
    #     bvp_posterior.locations[:-1][insert_two], bvp_posterior.locations[-1]
    # )

    # print(insert_one_points, insert_two_points)

    # print(insert_one.shape)
    # print(insert_two.shape)
    # print(bvp_posterior.locations.shape)
    # print(insert_one_points.shape)
    # print(insert_two_points.shape)
    # print()

    # assert False

    a1, *_ = insert_central_point(bvp_posterior.locations, insert_one)
    a2, *_ = insert_two_equispaced_points(bvp_posterior.locations, insert_two)

    new_mesh = np.union1d(a1, a2)
    return (
        new_mesh,
        integral_error,
        quotient,
        new_candidates,
        dt,
        insert_one,
        insert_two,
    )
