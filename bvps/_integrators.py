"""Integrators that are used for BVPs!"""

import numpy as np
from probnum import statespace
from ._problems import SecondOrderBoundaryValueProblem

__all__ = ["WrappedIntegrator"]


SMALL_VALUE = 0.0


class WrappedIntegrator(statespace.Integrator, statespace.LTISDE):
    """
    Examples
    --------
    >>> from probnum import statespace
    >>> from probnum import random_variables as randvars
    >>> from bvps import bratus, BoundaryValueProblem
    >>> bvp = bratus()
    >>> print(isinstance(bvp, BoundaryValueProblem))
    True
    >>> ibm = statespace.IBM(ordint=2, spatialdim=2)
    >>> print(isinstance(ibm, statespace.Integrator))
    True

    >>> integ = WrappedIntegrator(ibm, bvp)
    >>> print(integ)
    <WrappedIntegrator object>
    >>> rv = randvars.Normal(np.ones(ibm.dimension), np.eye(ibm.dimension))
    >>> out, _  = integ.forward_realization(rv.sample(), bvp.t0, 0.1)
    >>> print(out)
    <Normal with shape=(6,), dtype=float64>
    """

    def __init__(self, integrator, bvp):
        self.integrator = integrator
        self.bvp = bvp

        L, R = self.bvp.L, self.bvp.R

        if isinstance(bvp, SecondOrderBoundaryValueProblem):

            proj = np.stack(
                (self.integrator.proj2coord(0)[0], self.integrator.proj2coord(1)[0])
            )
        else:
            proj = self.integrator.proj2coord(0)
        # print(proj.shape, R.shape)
        Rnew = R @ proj
        Lnew = L @ proj

        self.measmod_R = statespace.DiscreteLTIGaussian(
            Rnew,
            -self.bvp.ymax,
            SMALL_VALUE * np.eye(len(R)),
            proc_noise_cov_cholesky=np.sqrt(SMALL_VALUE) * np.eye(len(R)),
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )
        self.measmod_L = statespace.DiscreteLTIGaussian(
            Lnew,
            -self.bvp.y0,
            SMALL_VALUE * np.eye(len(L)),
            proc_noise_cov_cholesky=np.sqrt(SMALL_VALUE) * np.eye(len(L)),
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )

        # print(self.measmod_L.input_dim)
        # print(self.measmod_L.output_dim)

    def __repr__(self):
        return "<WrappedIntegrator object>"

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):

        if np.abs(dt) == 0.0:
            assert t == self.bvp.t0

            rv, _ = self._update_rv_initial_value(rv, t)
            # return rv, _

        if np.abs(dt) > 0.0:
            # Plain old forward rv
            rv, _ = self.integrator.forward_rv(rv, t, dt=dt)

        rv, _ = self._update_rv_final_value(rv, t + dt)
        return rv, {}

    def _update_rv_final_value(self, rv, t):

        """Update a random variable on final and initial values."""

        dt_tmax = self.bvp.tmax - t

        if np.abs(dt_tmax) > 0.0:
            # Extrapolate to end point
            final_point_rv, info = self.integrator.forward_rv(rv, t, dt=dt_tmax)

        else:
            assert t == self.bvp.tmax

            final_point_rv = rv

        # Condition on measurement at endpoint
        zero_data = np.zeros(len(self.bvp.ymax))
        updated_final_point_rv, _ = self.measmod_R.backward_realization(
            realization_obtained=zero_data,
            rv=final_point_rv,
            t=self.bvp.tmax,
        )
        # print(updated_final_point_rv.mean)
        # print(self.bvp.ymax)
        # print(t)

        if np.abs(dt_tmax) > 0.0:
            # print(np.linalg.norm(updated_final_point_rv.cov_cholesky))

            # Condition back to plain old forwarded rv
            updated_rv, _ = self.integrator.backward_rv(
                rv_obtained=updated_final_point_rv, rv=rv, t=t, dt=dt_tmax
            )
        else:
            updated_rv = updated_final_point_rv
        return updated_rv, {}

    def _update_rv_initial_value(self, rv, t):

        dt_t0 = self.bvp.t0 - t

        if np.abs(dt_t0) > 0.0:
            raise RuntimeError
            # Extrapolate to initial point
            final_point_rv, info = self.integrator.forward_rv(rv, t, dt=dt_t0)
        else:
            final_point_rv = rv

        # Condition on measurement at endpoint
        zero_data = np.zeros(len(self.bvp.y0))
        updated_final_point_rv, _ = self.measmod_L.backward_realization(
            realization_obtained=zero_data,
            rv=final_point_rv,
            t=self.bvp.t0,
        )

        if np.abs(dt_t0) > 0.0:
            raise RuntimeError
            # Condition back to plain old forwarded rv
            updated_rv, _ = self.integrator.backward_rv(
                rv_obtained=updated_final_point_rv, rv=rv, t=t, dt=dt_t0
            )
        else:
            updated_rv = updated_final_point_rv

        return updated_rv, {}

    def backward_rv(self, *args, **kwargs):
        return self.integrator.backward_rv(*args, **kwargs)

    def backward_realization(self, *args, **kwargs):
        return self.integrator.backward_realization(*args, **kwargs)

    @property
    def ordint(self):
        return self.integrator.ordint

    @property
    def spatialdim(self):
        return self.integrator.ordint

    def proj2coord(self, *args, **kwargs):
        return self.integrator.proj2coord(*args, **kwargs)

    @property
    def dimension(self):
        return self.integrator.dimension

    @property
    def spatialdim(self):
        return self.integrator.spatialdim
