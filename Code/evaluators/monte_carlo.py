"Classes for running Monte Carlo Simulations"

import numpy as np
import pandas as pd
from IPython.display import display

class MonteCarlo:
    def __init__(self, model):
        self.model = model

    def _generate_path(self, r0, T: float, m: int, n=None, P=None, J=None, J_pos=None, model=None):
        """
        r0, float: i-rate today
        T, float: Maturity date (in years)
        m, int: time steps per year,
        n, 1x(T*m) np.array, array of draws,
        """
        if model is None:
            model = self.model
        dt = 1/m
        num_steps = int(np.ceil(T/dt))
        r = np.empty(num_steps + 1)
        none = False
        if n is None:
            n = np.empty(num_steps)
            P = np.empty(num_steps)
            J = np.empty(num_steps)
            J_pos = np.empty(num_steps)
            none = True
        r[0] = r0
        for j in range(num_steps):
            r[j+1], n[j], P[j], J[j], J_pos[j] = model.increment(rj=r[j], dt=dt, 
                                                 nj=-n[j] if not none else None,
                                                 Pj=P[j] if not none else None,
                                                 Jj=-J[j] if not none else None,
                                                 Jj_pos=J_pos[j] if not none else None
                                                )
        return r, n, P, J, J_pos
    def _evaluate_price(self, r: np.array, m: int):
        """
        r, 1xk np.array: i-rate path,
        m, int: time steps per year
        """
        dt = 1/m
        return np.exp(-np.sum(np.multiply(r, dt)))

    def _simulate_paths(self, m: int, r0: float, n: int, T: float):
        """
        m, int: time steps per year
        r0, float: i-rate today
        n, int: number of simulations
        T, float: Maturity date (in years)
        """
        prices = np.empty(n)
        for i in range(n):
            r, _, __, ___, ____ = self._generate_path(r0, T, m)
            prices[i] = self._evaluate_price(r, m)
        return np.mean(prices), np.std(prices)

    def _simulate_paths_anti(self, m: int, r0: float, n: int, T: float):
        """
        m, int: time steps per year
        r0, float: i-rate today
        n, int: number of simulations
        T, float: Maturity date (in years)
        """
        prices = np.empty(n)
        for i in range(n):
            r1, n, P, J, J_pos = self._generate_path(r0, T, m)
            r2, _, __, ___, ____ = self._generate_path(r0, T, m, n, P, J, J_pos)
            prices[i] = 0.5*(self._evaluate_price(r1, m) + self._evaluate_price(r2, m))
        return np.mean(prices), np.std(prices)

    def _simulate_paths_cv(self, m: int, r0: float, n: int, T: float, exact_model_class):
        """
        m, int: time steps per year
        r0, float: i-rate today
        n, int: number of simulations
        T, float: Maturity date (in years)
        exact_model_class, i-rate model class with exact solution
        """
        exact_model = exact_model_class(self.model.model_params)
        prices = np.empty(n)
        prices_classical = np.empty(n)
        prices_jump = np.empty(n)
        exact_price = exact_model.exact(r0=r0, T=T)
        for i in range(n):
            r_exact, norm, P, J, J_pos = self._generate_path(r0, T, m, model=exact_model)
            r_actual, _ , __, ___, ____ = self._generate_path(r0, T, m, np.negative(norm), P, np.negative(J), np.logical_not(J_pos))
            prices_classical[i] = self._evaluate_price(r_exact, m)
            prices_jump[i] = self._evaluate_price(r_actual, m)        
        c = -np.cov(prices_classical, prices_jump)[0][1]/np.var(prices_classical)
        prices = prices_jump + np.multiply(c, prices_classical - exact_price)
        return np.mean(prices), np.std(prices)
    
    def _simulate_paths_cv_anti(self, m: int, r0: float, n: int, T: float, exact_model_class):
        """
        m, int: time steps per year
        r0, float: i-rate today
        n, int: number of simulations
        T, float: Maturity date (in years)
        exact_model_class, i-rate model class with exact solution
        """
        exact_model = exact_model_class(self.model.model_params)
        prices = np.empty(n)
        prices_classical = np.empty(n)
        prices_jump = np.empty(n)
        exact_price = exact_model.exact(r0=r0, T=T)
        for i in range(n):
            r_exact, norm, P, J, J_pos = self._generate_path(r0, T, m, model=exact_model)
            r2_exact, _ , __, ___, ____ = self._generate_path(r0, T, m, norm, P, J, J_pos, model=exact_model)
            prices_classical[i] = 0.5*(self._evaluate_price(r_exact, m) + self._evaluate_price(r2_exact, m)) 
            r_actual, norm, P, J, J_pos = self._generate_path(r0, T, m, np.negative(norm), P, np.negative(J), np.logical_not(J_pos))
            r2_actual, _ , __, ___, ____ = self._generate_path(r0, T, m, norm, P, J, J_pos)
            prices_jump[i] = 0.5*(self._evaluate_price(r_actual, m) + self._evaluate_price(r2_actual, m))       
        c = -np.cov(prices_classical, prices_jump)[0][1]/np.var(prices_classical)
        prices = prices_jump + np.multiply(c, prices_classical - exact_price)
        return np.mean(prices), np.std(prices)