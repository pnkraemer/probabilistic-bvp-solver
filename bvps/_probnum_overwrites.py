"""These objects are suitable replacements of the corresponding ProbNum objects."""

from probnum import statespace, filtsmooth
from probnum import random_variables as randvars
import numpy as np
import scipy.linalg
from probnum._randomvariablelist import _RandomVariableList

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

    def iterated_filtsmooth(self, dataset, times, stopcrit=None, old_posterior=None):
        """Compute an iterated smoothing estimate with repeated posterior linearisation.

        If the extended Kalman filter is used, this yields the IEKS. In
        any case, the result is an approximation to the maximum-a-
        posteriori estimate.
        """

        if stopcrit is None:
            stopcrit = MyStoppingCriterion()

        # Initialise iterated smoother
        if old_posterior is None:
            old_posterior = self.filtsmooth(
                dataset=dataset,
                times=times,
                _previous_posterior=None,
            )
        new_posterior = old_posterior
        new_mean = new_posterior.state_rvs.mean
        old_mean = np.inf * np.ones(new_mean.shape)
        errors = np.inf * np.ones(new_mean.shape)
        while not stopcrit.terminate(error=errors, reference=new_mean):
            old_posterior = new_posterior
            new_posterior = self.filtsmooth(
                dataset=dataset,
                times=times,
                _previous_posterior=old_posterior,
            )


            msrvs = _RandomVariableList(
                [
                    self.measurement_model.forward_realization(
                        x.mean, t=t
                    )[0]
                    for t, x in zip(new_posterior.locations, new_posterior.state_rvs)
                ]
            )
            errors = np.abs(msrvs.mean)
            # new_mean = new_posterior.state_rvs.mean
            new_mean = np.ones((len(msrvs), len(msrvs[0].mean)))




            old_mean = old_posterior(new_posterior.locations).mean

            # errors = new_mean - old_mean
            print(stopcrit.evaluate_error(errors, new_mean))


        return new_posterior

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


class MyStoppingCriterion(filtsmooth.StoppingCriterion):
    def __init__(self, atol=1e-3, rtol=1e-6, maxit=1000, maxit_reached="error"):
        self.atol = atol
        self.rtol = rtol
        self.maxit = maxit
        self.iterations = 0

        def error(msg):
            raise RuntimeError(msg)

        def warning(msg):
            print(msg)

        def go_on(*args, **kwargs):
            pass

        options = {
            "error": error,
            "warning": warning,
            "pass": go_on,
        }
        self.maxit_behaviour = options[maxit_reached]

        self.previous_number_of_iterations = 0

    def terminate(self, error, reference):
        """Decide whether the stopping criterion is satisfied, which implies terminating
        of the iteration.

        If the error is sufficiently small (with respect to atol, rtol
        and the reference), return True. Else, return False. Throw a
        runtime error if the maximum number of iterations is reached.
        """
        if self.iterations > self.maxit:
            errormsg = f"Maximum number of iterations (N={self.maxit}) reached."
            self.maxit_behaviour(errormsg)

        magnitude = self.evaluate_error(error=error, reference=reference)
        # print("M", magnitude)
        if magnitude > 1:
            self.iterations += 1
            return False
        else:
            self.previous_number_of_iterations = self.iterations
            self.iterations = 0
            return True

    def evaluate_error(self, error, reference):
        """Compute the normalised error."""
        normalisation = self.atol + self.rtol * reference

        # return np.amax(error / normalisation)
        magnitude = np.sqrt(np.mean((error / normalisation) ** 2))
        return magnitude



class MyIteratedDiscreteComponent(filtsmooth.IteratedDiscreteComponent):


    def backward_rv(
        self,
        rv_obtained,
        rv,
        rv_forwarded=None,
        gain=None,
        t=None,
        dt=None,
        _diffusion=1.0,
        _linearise_at=None,
    ):
        current_rv, info = self._component.backward_rv(
            rv_obtained=rv_obtained,
            rv=rv,
            t=t,
            dt=dt,
            _diffusion=_diffusion,
            _linearise_at=_linearise_at,
        )

        new_mean = current_rv.mean.copy()
        old_mean = np.inf * np.ones(current_rv.mean.shape)



        reference = np.ones(1)
        error = np.inf * np.ones(1)

        while not self.stopcrit.terminate(
            error=error, reference=reference
        ):
            old_mean = new_mean.copy()
            current_rv, info = self._component.backward_rv(
                rv_obtained=rv_obtained,
                rv=rv,
                t=t,
                dt=dt,
                _diffusion=_diffusion,
                _linearise_at=current_rv,
            )

            fwd, _ = self._component.forward_realization(current_rv.mean, t=t)
            error = np.abs(fwd.mean)
            reference = np.ones(error.shape)

            new_mean = current_rv.mean.copy()

        return current_rv, info
