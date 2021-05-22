"""Integrated bridges."""

import numpy as np
from probnum import statespace

from .ode_measmods import from_boundary_conditions
from .problems import SecondOrderBoundaryValueProblem

SMALL_VALUE = 0.0


class GaussMarkovBridge(statespace.Integrator, statespace.LTISDE):
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

    >>> integ = GaussMarkovBridge(ibm, bvp)
    >>> print(integ)
    <GaussMarkovBridge object>
    >>> rv = randvars.Normal(np.ones(ibm.dimension), np.eye(ibm.dimension))
    >>> out, _  = integ.forward_realization(rv.sample(), bvp.t0, 0.1)
    >>> print(out)
    <Normal with shape=(6,), dtype=float64>
    """

    def __init__(self, integrator, bvp):
        self.integrator = integrator
        self.bvp = bvp
        self.measmod_L, self.measmod_R = from_boundary_conditions(
            bvp=bvp, prior=integrator
        )

        # if isinstance(bvp, SecondOrderBoundaryValueProblem):
        #     proj = np.stack(
        #         (self.integrator.proj2coord(0)[0], self.integrator.proj2coord(1)[0])
        #     )
        # else:
        #     proj = self.integrator.proj2coord(0)

        # Rnew = R @ proj
        # Lnew = L @ proj

        # self.measmod_R = statespace.DiscreteLTIGaussian(
        #     Rnew,
        #     -self.bvp.ymax,
        #     SMALL_VALUE * np.eye(len(R)),
        #     proc_noise_cov_cholesky=np.sqrt(SMALL_VALUE) * np.eye(len(R)),
        #     forward_implementation="sqrt",
        #     backward_implementation="sqrt",
        # )
        # self.measmod_L = statespace.DiscreteLTIGaussian(
        #     Lnew,
        #     -self.bvp.y0,
        #     SMALL_VALUE * np.eye(len(L)),
        #     proc_noise_cov_cholesky=np.sqrt(SMALL_VALUE) * np.eye(len(L)),
        #     forward_implementation="sqrt",
        #     backward_implementation="sqrt",
        # )

        self.boundary_conditions_initialised = False

    def __repr__(self):
        return "<GaussMarkovBridge object>"

    def initialise_boundary_conditions(self, initrv):
        if self.boundary_conditions_initialised:
            raise RuntimeError("Already done")
        self.boundary_conditions_initialised = True

        initrv_updated, _ = self._update_rv_initial_value(initrv)
        finalrv, _ = self._update_rv_final_value(initrv_updated, self.bvp.t0)

        return finalrv

    def forward_rv(
        self,
        rv,
        t,
        dt=None,
        compute_gain=False,
        _diffusion=1.0,
        **kwargs,
    ):
        if not self.boundary_conditions_initialised:
            raise RuntimeError("BDRY COND.")

        if not dt > 0.0:
            raise RuntimeError("STEP")

        if np.abs(dt) > 0.0:
            finalrv, _ = self.integrator.forward_rv(rv, t, dt=dt)

        forwarded_rv, _ = self._update_rv_final_value(finalrv, t + dt)
        return forwarded_rv, {}

    def _update_rv_final_value(self, rv, t):
        """Update a random variable on final and initial values."""

        dt_tmax = self.bvp.tmax - t

        # Extrapolate to end point
        if np.abs(dt_tmax) > 0.0:
            final_point_rv, info = self.integrator.forward_rv(rv, t, dt=dt_tmax)
        else:
            assert t == self.bvp.tmax
            final_point_rv = rv

        # Condition on measurement at endpoint
        zero_data = np.zeros(self.measmod_R.output_dim)
        updated_final_point_rv, _ = self.measmod_R.backward_realization(
            realization_obtained=zero_data,
            rv=final_point_rv,
            t=self.bvp.tmax,
        )

        # Pass info back to initial rv (i.e. before forwardining to endpoint)
        if np.abs(dt_tmax) > 0.0:
            updated_rv, _ = self.integrator.backward_rv(
                rv_obtained=updated_final_point_rv, rv=rv, t=t, dt=dt_tmax
            )
        else:
            updated_rv = updated_final_point_rv

        return updated_rv, {}

    def _update_rv_initial_value(self, rv):

        zero_data = np.zeros(self.measmod_L.output_dim)
        updated_rv, _ = self.measmod_L.backward_realization(
            realization_obtained=zero_data,
            rv=rv,
            t=self.bvp.t0,
        )

        return updated_rv, {}

    def backward_rv(self, *args, **kwargs):
        return self.integrator.backward_rv(*args, **kwargs)

    def backward_realization(self, *args, **kwargs):
        return self.integrator.backward_realization(*args, **kwargs)

    def proj2coord(self, *args, **kwargs):
        return self.integrator.proj2coord(*args, **kwargs)

    @property
    def ordint(self):
        return self.integrator.ordint

    @property
    def dimension(self):
        return self.integrator.dimension

    @property
    def spatialdim(self):
        return self.integrator.spatialdim
