"""Classes for CIR Model."""

import math
import numpy as np
from scipy.stats import norm

class CIR:
    """Canonical CIR Model."""
    def __init__(self, model_params: dict):
        self.model_params = model_params
    def increment(self, rj: float, dt: float, nj=None, Pj=None, Jj=None, Jj_pos=None):
        """Calculates next interest rate step."""
        if nj is None or Pj is None: 
            nj = np.random.normal()
            Pj = np.random.poisson(self.model_params["h"]*dt)
        time_step = self.model_params["kappa"]*(self.model_params["mu_r"] - rj)*dt
        stoch_step = self.model_params["sigma"]*np.sqrt(np.abs(rj))*np.sqrt(dt)*nj
        return rj + time_step + stoch_step, nj, Pj, None, None
    
    def exact(self, r0, T):
        """Returns exact price rate for maturity T."""
        K_hat = self.model_params["kappa"]
        mu_hat = self.model_params["mu_r"]
        sigma = self.model_params['sigma']
        gamma = np.sqrt(K_hat**2 + 2*sigma**2)
        B = 2*(np.exp(gamma*T) - 1) / ((gamma + K_hat)*(np.exp(gamma*T)-1) + 2*gamma)
        A = (2*K_hat*mu_hat / sigma**2) * (np.log(2*gamma*np.exp((gamma+K_hat)*T*0.5)) - np.log((gamma+K_hat)*(np.exp(gamma*T) - 1) + 2*gamma))
        return np.exp(A - B*r0)
    
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
        for n in range(1):  # Note we only iterate once for non-jump models
            normal_sd = np.sqrt(dt*(rt_1*(sigma**2)))
            normal_mean = rt_1 + kappa*(mu_r - rt_1)*dt
            normal_density = norm.pdf(rt, loc=normal_mean, scale=normal_sd)
            sum_ += normal_density
        return sum_
