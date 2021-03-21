"""These objects are suitable replacements of the corresponding ProbNum objects."""

from probnum import statespace, filtsmooth
from probnum import random_variables as randvars
import numpy as np
import scipy.linalg

__all__ = ["from_ode", "MyKalman"]


def from_ode(ode, prior):

    spatialdim = prior.spatialdim
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)

    def dyna(t, x):
        return h1 @ x - ode.f(t, h0 @ x)

    def diff_cholesky(t):
        return 0.0 * np.eye(spatialdim)

    def jacobian(t, x):
        return h1 - ode.df(t, h0 @ x) @ h0

    discrete_model = statespace.DiscreteGaussian(
        input_dim=prior.dimension,
        output_dim=spatialdim,
        state_trans_fun=dyna,
        proc_noise_cov_mat_fun=diff_cholesky,
        jacob_state_trans_fun=jacobian,
        proc_noise_cov_cholesky_fun=diff_cholesky,
    )
    return filtsmooth.DiscreteEKFComponent(
        discrete_model,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )


class MyKalman(filtsmooth.Kalman):
    """Kalman filtering with calibration"""

    def filter(
        self,
        dataset: np.ndarray,
        times: np.ndarray,
        _previous_posterior=None,
    ):
        """Apply Gaussian filtering (no smoothing!) to a data set.

        Parameters
        ----------
        dataset : array_like, shape (N, M)
            Data set that is filtered.
        times : array_like, shape (N,)
            Temporal locations of the data points.
            The zeroth element in times and dataset is the location of the initial random variable.
        _previous_posterior: KalmanPosterior
            If specified, approximate Gaussian filtering and smoothing linearises at this, prescribed posterior.
            This is used for iterated filtering and smoothing. For standard filtering, this can be ignored.

        Returns
        -------
        KalmanPosterior
            Posterior distribution of the filtered output
        """
        dataset, times = np.asarray(dataset), np.asarray(times)
        rvs = []
        sigmas = []

        _linearise_update_at = (
            None if _previous_posterior is None else _previous_posterior(times[0])
        )
        filtrv, *_ = self.update(
            data=dataset[0],
            rv=self.initrv,
            time=times[0],
            _linearise_at=_linearise_update_at,
        )

        rvs.append(filtrv)
        for idx in range(1, len(times)):
            _linearise_predict_at = (
                None
                if _previous_posterior is None
                else _previous_posterior(times[idx - 1])
            )
            _linearise_update_at = (
                None if _previous_posterior is None else _previous_posterior(times[idx])
            )

            filtrv, info = self.filter_step(
                start=times[idx - 1],
                stop=times[idx],
                current_rv=filtrv,
                data=dataset[idx],
                _linearise_predict_at=_linearise_predict_at,
                _linearise_update_at=_linearise_update_at,
            )

            sigma = info["info_upd"]["current_sigma"]
            sigmas.append(sigma)
            # print(sigma)

            rvs.append(filtrv)

        ssq = np.mean(sigmas)
        # rvs = [
        #     randvars.Normal(
        #         mean=rv.mean,
        #         cov=ssq * rv.cov,
        #         cov_cholesky=np.sqrt(ssq) * rv.cov_cholesky,
        #     )
        #     for rv in rvs
        # ]
        # print("Warning: what about IEKF with the update?")
        # print("global sigma", ssq)
        self.ssq = ssq
        # print()
        return filtsmooth.FilteringPosterior(times, rvs, self.dynamics_model)

    def update(self, rv, time, data, _linearise_at=None):

        meas_rv, _ = self.measurement_model.forward_rv(
            rv, t=time, _linearise_at=_linearise_at, compute_gain=True
        )
        # upd_rv, info2 = self.measurement_model.backward_realization(
        #     data,
        #     rv,
        #     t=time,
        #     _linearise_at=_linearise_at,
        #     forwarded_rv=meas_rv,
        #     gain=info["gain"],
        # )
        upd_rv, info = self.measurement_model.backward_realization(
            data, rv, t=time, _linearise_at=_linearise_at
        )

        sigma = meas_rv.mean.T @ scipy.linalg.cho_solve(
            (meas_rv.cov_cholesky, True), meas_rv.mean
        )
        info["current_sigma"] = sigma

        return upd_rv, info
