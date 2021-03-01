"""Classes for Jump CIR Model."""

import math
import numpy as np
from scipy.stats import truncnorm, norm, poisson

class JumpCIR:
    """CIR Model with jumps."""
    def __init__(self, model_params: dict):
        self.model_params = model_params
    def increment(self, rj: float, dt: float, nj=None, Pj=None, Jj=None, Jj_pos=None):
        """Calculates next interest rate step."""
        if nj is None or Pj is None: 
            nj = np.random.normal()
            Pj = np.random.poisson(self.model_params["h"]*dt)
        time_step = self.model_params["kappa"]*(self.model_params["mu_r"] - rj)*dt
        stoch_step = self.model_params["sigma"]*np.sqrt(np.abs(rj))*np.sqrt(dt)*nj
        if Jj is None or math.isnan(Jj):
            Jj = self.jump_norm(rj, dt, nj, Pj, time_step, stoch_step)
            if Jj_pos is not None:
                if Jj_pos and Jj > 0:
                    Jj = -Jj
                if not Jj_pos and Jj < 0:
                    Jj = -Jj

        jump_step = Jj*Pj

        return rj + time_step + stoch_step + jump_step, nj, Pj, None, Jj > 0
    
    def jump_norm(self, rj: float, dt: float, nj: float, Pj: float, time_step: float, stoch_step: float):
        """Calculates truncated jump for CIR."""
        if Pj == 0:
            Jj = 0
        else:
            lower_bound = (-rj - time_step - stoch_step)/Pj
            try:
                Jj = truncnorm(-np.abs(lower_bound), np.abs(lower_bound), loc=self.model_params["mu"], scale=self.model_params["gamma"]).rvs(1)[0]
            except Exception as de:
                print("Domain Error")
                print(f"Param values: {rj, dt, nj, Pj, time_step, stoch_step}")
        return Jj

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
        kappa, mu_r, sigma, mu, gamma, h = self.model_params["kappa"], self.model_params["mu_r"], self.model_params["sigma"], self.model_params["mu"], self.model_params["gamma"], self.model_params["h"]
        sum_ = 0
        for n in range(limit):
            expon_density = poisson.pmf(n, mu=h*dt)
            normal_sd = np.sqrt(n*(gamma**2) + dt*(rt_1*(sigma**2)))
            normal_mean = rt_1 + kappa*(mu_r - rt_1)*dt + mu*n
            normal_density = norm.pdf(rt, loc=normal_mean, scale=normal_sd)
            sum_ += normal_density*expon_density
        return sum_
