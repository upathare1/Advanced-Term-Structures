"Classes for Jump Vasicek Model"

import numpy as np
import pandas as pd

class Vasicek:
    """Canonical Vasicek Model"""
    def __init__(self, model_params: dict):
        self.model_params = model_params
    def increment(self, rj, dt, nj=None, Pj=None, Jj=None):
        if nj is None:
            nj = np.random.normal()
            Pj = np.random.poisson(model_params["h"]*dt)
            Jj = np.random.normal(model_params["mu"], model_params["gamma"])
        time_step = self.model_params["kappa"]*(self.model_params["mu_r"] - rj)*dt
        stoch_step = self.model_params["sigma"]*np.sqrt(dt)*nj
        return rj + time_step + stoch_step, nj, Pj, Jj
    def exact(self, r0, T):
        """
        Returns exact price rate for maturity T
        """
        K = self.model_params["kappa"]
        u_hat = self.model_params["mu_r"]
        B = (1 - np.exp(-K*(T)))/K
        A = (B - (T))*(u_hat - self.model_params["sigma"]**2/(2*K*K)) - (self.model_params["sigma"]**2)*(B**2)/(4*K)
        return np.exp(A - B*r0)