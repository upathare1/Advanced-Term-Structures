"Classes for Jump Vasicek Model"

import math
import numpy as np
from scipy.stats import norm, poisson


class JumpVasicek:
    """Vasicek Model with jumps"""
    def __init__(self, model_params: dict):
        self.model_params = model_params

    def increment(self,
                  rj: float,
                  dt: float,
                  nj=None,
                  Pj=None,
                  Jj=None,
                  Jj_pos=None):
        """
        Calculates next interest rate step
        """
        if nj is None or Pj is None or Jj is None:
            nj = np.random.normal()
            Pj = np.random.poisson(self.model_params["h"] * dt)
            Jj = np.random.normal(self.model_params["mu"],
                                  self.model_params["gamma"])
        time_step = self.model_params["kappa"] * (self.model_params["mu_r"] - rj) * dt
        stoch_step = self.model_params["sigma"] * np.sqrt(dt) * nj
        jump_step = Jj * Pj
        return rj + time_step + stoch_step + jump_step, nj, Pj, Jj, Jj_pos

    def transition(self, rt, rt_1, dt, limit=2) -> float:
        """
        Calculate the transition density for rt for an observed rt_1 and model parameters.
        Calculate the first (limit) terms of the infinite series
        --
        rt, float: observed rate at t,
        rt_1, float: observed rate at t-dt,
        dt, float: time step,
        limit, int: number of terms in the inf series to calculate
        """
        kappa, mu_r, sigma, gamma, h = self.model_params["kappa"], self.model_params["mu_r"], self.model_params["sigma"], self.model_params["gamma"], self.model_params["h"]
        sum_ = 0
        for n in range(limit):
            expon_density = poisson.pmf(n, mu=h*dt)
            normal_sd = np.sqrt(n*(gamma**2) + dt*(sigma**2))
            normal_mean = rt_1 + kappa*(mu_r - rt_1)*dt
            normal_density = norm.pdf(rt, loc=normal_mean, scale=normal_sd)
            sum_ += normal_density*expon_density
        return sum_
