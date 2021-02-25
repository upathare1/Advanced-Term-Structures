"Classes for Jump Vasicek Model"

import numpy as np
import pandas as pd

class Vasicek:
    """Canonical Vasicek Model"""
    def __init__(self, model_params: dict):
        self.model_params = model_params
    def increment(self, rj, dt, nj=None, Pj=None, Jj=None, Jj_pos=None):
        if nj is None:
            nj = np.random.normal()
            Pj = np.random.poisson(self.model_params["h"]*dt)
            Jj = np.random.normal(self.model_params["mu"], self.model_params["gamma"])
        time_step = self.model_params["kappa"]*(self.model_params["mu_r"] - rj)*dt
        stoch_step = self.model_params["sigma"]*np.sqrt(dt)*nj
        return rj + time_step + stoch_step, nj, Pj, Jj, Jj_pos
    def exact(self, r0, T):
        """
        Returns exact price rate for maturity T
        """
        K = self.model_params["kappa"]
        u_hat = self.model_params["mu_r"]
        B = (1 - np.exp(-K*(T)))/K
        A = (B - (T))*(u_hat - self.model_params["sigma"]**2/(2*K*K)) - (self.model_params["sigma"]**2)*(B**2)/(4*K)
        return np.exp(A - B*r0)
    
    def transition(self, rt, rt_1, dt, limit=2):
        """
        Calculate the transition density for rt for an observed rt_1 and model parameters.
        Calculate the first (limit) terms of the infinite series
        --
        rt, float: observed rate at t,
        rt_1, float: observed rate at t-dt,
        dt, float: time step,
        limit, int: number of terms in the inf series to calculate
        """
        kappa, mu_r, sigma, mu, gamma, h = self.model_params["kappa"], self.model_params["mu_r"], self.model_params["sigma"], self.model_params['mu'], self.model_params["gamma"], self.model_params["h"]
        sum_ = 0
        for n in range(limit):
            normal_sd = np.sqrt(dt*(sigma**2))
            normal_mean = rt_1 + kappa*(mu_r - rt_1)*dt
            normal_density = norm.pdf(rt, loc=normal_mean, scale=normal_sd)
            sum_ += normal_density
        return sum_