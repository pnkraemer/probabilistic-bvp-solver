"""Sampling from a prior."""


def generate_samples(grid, transition, rv, base_measure_samples):
    smp = rv.mean + rv.cov_cholesky @ base_measure_samples[0]
    yield smp
    for t, tnew, b in zip(grid[:-1], grid[1:], base_measure_samples[1:]):
        dt = tnew - t
        rv, _ = transition.forward_realization(smp, t=t, dt=dt)
        smp = rv.mean + rv.cov_cholesky @ b
        yield smp
