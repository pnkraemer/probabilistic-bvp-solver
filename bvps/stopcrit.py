"""Custom stopping criteria."""


import numpy as np
import scipy.linalg
from probnum import filtsmooth, randvars, statespace, utils
from probnum._randomvariablelist import _RandomVariableList


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
