"""These objects are suitable replacements of the corresponding ProbNum objects."""

import numpy as np
import scipy.linalg
from probnum import filtsmooth, statespace, utils, randvars
from probnum._randomvariablelist import _RandomVariableList


class MyKalman(filtsmooth.Kalman):
    """Kalman filtering with calibration"""

    def iterated_filtsmooth(
        self, dataset, times, measmodL, measmodR, stopcrit=None, old_posterior=None
    ):
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
                measmodL=measmodL,
                measmodR=measmodR,
                _previous_posterior=None,
            )
        new_posterior = old_posterior
        new_mean = new_posterior.states.mean
        old_mean = np.inf * np.ones(new_mean.shape)
        errors = np.inf * np.ones(new_mean.shape)
        while not stopcrit.terminate(error=errors, reference=new_mean):
            old_posterior = new_posterior
            new_posterior = self.filtsmooth(
                dataset=dataset,
                times=times,
                measmodR=measmodR,
                measmodL=measmodL,
                _previous_posterior=old_posterior,
            )

            new_initrv = new_posterior[0]

            new_mean = new_initrv.mean 
            new_cov_cholesky = utils.linalg.cholesky_update(new_initrv.cov_cholesky, new_mean - self.initrv.mean)
            new_cov = new_cov_cholesky @ new_cov_cholesky.T
            self.initrv = randvars.Normal(mean=new_mean, cov=new_cov, cov_cholesky=new_cov_cholesky)

            msrvs = _RandomVariableList(
                [
                    self.measurement_model.forward_realization(x.mean, t=t)[0]
                    for t, x in zip(new_posterior.locations, new_posterior.states)
                ]
            )
            errors = np.abs(msrvs.mean)
            # print(errors)
            new_mean = new_posterior.states.mean @ self.dynamics_model.proj2coord(1).T
            # new_mean = np.ones((len(msrvs), len(msrvs[0].mean)))
            # print(errors)
            old_mean = old_posterior(new_posterior.locations).mean

            # errors = new_mean - old_mean
            # print(stopcrit.evaluate_error(errors, new_mean))

        return new_posterior

    def filtsmooth(
        self,
        dataset: np.ndarray,
        times: np.ndarray,
        measmodL=None,
        measmodR=None,
        _previous_posterior=None,
    ):
        """Apply Gaussian filtering and smoothing to a data set.

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
        filter_posterior = self.filter(
            dataset,
            times,
            measmodL,
            measmodR,
            _previous_posterior=_previous_posterior,
        )
        smooth_posterior = self.smooth(filter_posterior)
        return smooth_posterior

    def filter(
        self,
        dataset: np.ndarray,
        times: np.ndarray,
        measmodL=None,
        measmodR=None,
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
        # print(times[0])
        _linearise_update_at = (
            None if _previous_posterior is None else _previous_posterior(times[0])
        )
        # if _previous_posterior is not None:
        #     self.initrv.mean = _previous_posterior[0].mean

        if measmodL is not None:
            filtrv, _ = measmodL.backward_realization(
                realization_obtained=dataset[0], rv=self.initrv, t=times[0]
            )
        else:
            filtrv = self.initrv

        filtrv, *_ = self.update(
            data=dataset[0],
            rv=filtrv,
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

        if measmodR is not None:
            rvs[-1], _ = measmodR.backward_realization(
                realization_obtained=dataset[-1], rv=rvs[-1], t=times[-1]
            )

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
        return filtsmooth.FilteringPosterior(
            locations=times, states=rvs, transition=self.dynamics_model
        )

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

    def filter_step(
        self,
        start,
        stop,
        current_rv,
        data,
        _linearise_predict_at=None,
        _linearise_update_at=None,
        _diffusion=1.0,
    ):
        """A single filter step.

        Consists of a prediction step (t -> t+1) and an update step (at t+1).

        Parameters
        ----------
        start : float
            Predict FROM this time point.
        stop : float
            Predict TO this time point.
        current_rv : RandomVariable
            Predict based on this random variable. For instance, this can be the result
            of a previous call to filter_step.
        data : array_like
            Compute the update based on this data.
        _linearise_predict_at
            Linearise the prediction step at this RV. Used for iterated filtering and smoothing.
        _linearise_update_at
            Linearise the update step at this RV. Used for iterated filtering and smoothing.
        _diffusion
            Custom diffusion for the underlying Wiener process. Used in calibration.

        Returns
        -------
        RandomVariable
            Resulting filter estimate after the single step.
        dict
            Additional information provided by predict() and update().
            Contains keys `pred_rv`, `info_pred`, `meas_rv`, `info_upd`.
        """
        data = np.asarray(data)
        info = {}
        info["pred_rv"], info["info_pred"] = self.predict(
            rv=current_rv,
            start=start,
            stop=stop,
            _linearise_at=_linearise_predict_at,
            _diffusion=_diffusion,
        )

        filtrv, info["info_upd"] = self.update(
            rv=info["pred_rv"],
            time=stop,
            data=data,
            _linearise_at=_linearise_update_at,
        )
        return filtrv, info


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
        # normalisation = self.atol + self.rtol * np.abs(reference)
        quotient = self.evaluate_quotient(error=error, reference=reference)
        magnitude = np.sqrt(np.mean(quotient ** 2))
        return magnitude

    def evaluate_quotient(self, error, reference):
        normalisation = self.atol + self.rtol * np.abs(reference)
        return error / normalisation


class ConstantStopping(filtsmooth.StoppingCriterion):
    def terminate(self, error, reference):

        if self.iterations > self.maxit:
            self.previous_number_of_iterations = self.iterations

            self.iterations = 0
            return True

        self.iterations += 1
        return False

    def evaluate_quotient(self, error, reference):
        normalisation = self.atol + self.rtol * np.abs(reference)
        return error / normalisation


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

        while not self.stopcrit.terminate(error=error, reference=reference):
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
