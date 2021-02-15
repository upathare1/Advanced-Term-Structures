"Classes for CIR Model"

import numpy as np
import pandas as pd

class CIR:
    """Canonical CIR Model"""
    def __init__(self, model_params: dict):
        self.model_params = model_params
    def increment(self, rj, dt, nj=None, Pj=None, Jj=None, Jj_pos=None):
        if nj is None:
            nj = np.random.normal()
        time_step = self.model_params["kappa"]*(self.model_params["mu_r"] - rj)*dt
        stoch_step = self.model_params["sigma"]*np.sqrt(rj)*np.sqrt(dt)*nj
        return rj + time_step + stoch_step, nj, Pj, Jj, Jj_pos
    def exact(self, r0, T):
        """
        Returns exact price rate for maturity T
        """
        K_hat = self.model_params["kappa"]
        mu_hat = self.model_params["mu_r"]
        sigma = self.model_params['sigma']
        gamma = np.sqrt(K_hat**2 + 2*sigma**2)
        B = 2*(np.exp(gamma*T) - 1) / ((gamma + K_hat)*(np.exp(gamma*T)-1) + 2*gamma)
        A = (2*K_hat*mu_hat / sigma**2) * (np.log(2*gamma*np.exp((gamma+K_hat)*T*0.5)) - np.log((gamma+K_hat)*(np.exp(gamma*T) - 1) + 2*gamma))
        return np.exp(A - B*r0)