"""Sampling from a prior."""


__all__ = ["generate_samples"]


def generate_samples(grid, transition, rv, base_measure_samples, fix=True):
    if fix:
        rv, _ = transition.forward_rv(rv, grid[0], dt=0.0)
    smp = rv.mean + rv.cov_cholesky @ base_measure_samples[0]
    yield smp
    for t, tnew, b in zip(grid[:-1], grid[1:], base_measure_samples[1:]):
        dt = tnew - t
        rv, _ = transition.forward_realization(smp, t=t, dt=dt)
        smp = rv.mean + rv.cov_cholesky @ b
        yield smp
