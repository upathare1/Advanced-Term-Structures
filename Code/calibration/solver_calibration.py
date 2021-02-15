"""Modules for managing solver-based model calibration"""

from copy import copy
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import OptimizeResult, minimize
from optimparallel import minimize_parallel
from evaluators.monte_carlo import MonteCarlo


class Calibration:
    """Class for calibrating i-rate models"""
    def __init__(self, model_class, n: int, m: int, r0: float, model_params: dict, optimize_args: tuple, seed=93756826):
        """
        model_class, class: Class for creating an interest rate model,
        model_params, dict: Starting parameters for model,
        optimize_args, tuple: Tuple of model parameter names (ex. ("gamma", "mu")) to optimize
        """
        self.n = n
        self.m = m
        self.r0 = r0
        self.seed = seed
        self.model_class = model_class
        self.model_params = model_params
        self.optimize_args = optimize_args
    
    def _calculate_error(self, optimize_params: np.array, Ts: np.array, prices: np.array) -> float:
        """
        Computes error (wrt observed prices) for given values of optimize_params (corresponding to optimize_args)
        First value of optmize_params is always r0
        """
        model_params = copy(self.model_params)
        for arg, param in zip(self.optimize_args, optimize_params):
            model_params[arg] = param
        model = self.model_class(model_params)
        mc = MonteCarlo(model)
        errors = np.empty(len(Ts))
        i = 0
        for price, T in zip(prices, Ts):
            np.random.seed(self.seed) # set seed to minimize variation in results arising from different draws
            errors[i] = mc._simulate_paths_anti(m=self.m, r0=0.05, n=self.n, T=T)[0] - price
            i += 1
        return np.linalg.norm(errors)

    def calibrate(self, initial_values: tuple, Ts: np.array, prices: np.array, bounds: Bounds) -> OptimizeResult:
        """
        initial_values, np.array: same length as optimize_args,
        Ts, np.array: array of maturities to fit,
        prices, np.array: array of prices to fit (same length as Ts),
        bounds, scipy.optimize.Bounds or None: linear bounds on solution
        """
        error_function = lambda optimize_params: self._calculate_error(optimize_params, Ts=Ts, prices=prices) 
        optimal = minimize(error_function, initial_values, method="L-BFGS-B", bounds=bounds)
        return optimal