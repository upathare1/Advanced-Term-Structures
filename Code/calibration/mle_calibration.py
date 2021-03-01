"""Module for managing mle-based model calibration."""

from copy import copy
import traceback
import numpy as np
from scipy.optimize import minimize, Bounds, OptimizeResult

class Calibration:
    """Class for calibrating i-rate models with MLE."""
    def __init__(self, data, dt, model_class, initial_params):
        self.data = data
        self.dt = dt
        self.model_class = model_class
        self.initial_params = initial_params

    def _nlog_likelihood(self, params: tuple, *args) -> float:
        """
        Calculates the negative log likelihood for given parameter values
        --
        params, tuple: tuple of argument values\n
        *args: tuple of arguments names\n
        """
        model_params = copy(self.initial_params)
        i = 0
        for param in args:
            model_params[param] = params[i]
            i += 1
        model = self.model_class(model_params)
        loglikelihood = 0
        for i in range(len(self.data) - 1):
            loglikelihood += np.log(
                model.transition(
                    rt=self.data[i + 1],
                    rt_1=self.data[i],
                    dt=self.dt))
        print(f"Params: {model_params}\nNeg Log Likelihood: {-loglikelihood}")
        return -loglikelihood

    def calibrate(
        self, bounds: Bounds, params=("kappa", "mu_r", "sigma", "gamma", "h")
    ) -> OptimizeResult:
        """
        Calculate optimal parameter values
        --
        bounds, scipy.minimize.Bounds: upper and lower bounds for parameters,\n
        params, tuple: names of parameters to optimize\n
        """
        initial_value = ()
        for param in params:
            initial_value = initial_value + (self.initial_params[param], )
        return minimize(self._nlog_likelihood,
                        initial_value,
                        args=params,
                        method="L-BFGS-B",
                        bounds=bounds)
