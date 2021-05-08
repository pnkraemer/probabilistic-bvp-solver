"""Error estimates."""
import numpy as np
from probnum import diffeq, randvars, utils
from probnum._randomvariablelist import _RandomVariableList


def estimate_errors_via_std(
    bvp_posterior, kalman_posterior, grid, ssq, measmod, atol, rtol
):
    evaluated_posterior = bvp_posterior(grid)
    errors = evaluated_posterior.std  # * np.sqrt(ssq)
    reference = evaluated_posterior.mean
    assert errors.shape == reference.shape
    return errors, evaluated_posterior.mean


def estimate_errors_via_defect(
    bvp_posterior, kalman_posterior, grid, ssq, measmod, atol, rtol
):
    h = np.amax(np.abs(np.diff(grid)))
    # print(h)
    evaluated_kalman_posterior = kalman_posterior(grid)
    msrvs = _RandomVariableList(
        [
            measmod.forward_realization(m, t=t)[0]
            for m, t in zip(evaluated_kalman_posterior.mean, grid)
        ]
    )
    errors = np.abs(msrvs.mean)
    reference = (
        evaluated_kalman_posterior.mean @ kalman_posterior.transition.proj2coord(0).T
    )
    assert errors.shape == reference.shape
    return errors, reference


def estimate_errors_via_probabilistic_defect(
    bvp_posterior, kalman_posterior, grid, ssq, measmod, atol, rtol
):
    h = np.amin(np.abs(np.diff(grid)))
    evaluated_kalman_posterior = kalman_posterior(grid)
    msrvs = _RandomVariableList(
        [
            measmod.forward_rv(rv, t=t)[0]
            for rv, t in zip(evaluated_kalman_posterior, grid)
        ]
    )

    reference = (
        evaluated_kalman_posterior.mean @ kalman_posterior.transition.proj2coord(0).T
    )
    refs = atol + rtol * np.abs(reference)

    errors1 = np.abs(msrvs.mean) ** 2 / (refs ** 2)
    errors2 = np.abs(msrvs.std) ** 2 / (refs ** 2) * ssq * 0.0

    error_std = np.sum(errors2, axis=1)
    error_mean = np.sum(errors1, axis=1)
    quotient = error_mean  # * h

    assert errors1.shape == reference.shape
    return None, None, quotient
