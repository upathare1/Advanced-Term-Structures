"Classes for Jump Vasicek Model"

import numpy as np

class JumpVasicek: 
    """Vasiceck Spot-Rate Model with Jumps"""
    def __init__(self, model_params: dict):
        self.model_params = model_params

    def increment(self, rj, t, T, m, nj=None, pj=None, Jj=None):
        """
        Incrementor function for Jump Vasicek
        All incrementors will have nj, pj, and Jj arguments even if classical models dont need pj or Jj
        """
        delta_t = (T-t)/m
        theta = self.model_params["theta"]
        sigma = self.model_params["sigma"]
        a = self.model_params["a"]
        lambda_ = self.model_params["lambda"]
        h = self.model_params["h"]
        gamma = self.model_params["gamma"]
        mu = self.model_params["mu"]

        if nj is None or pj is None or Jj is None:
            nj = np.random.normal()
            pj = np.random.poisson(h*delta_t)
            Jj = np.random.normal(mu, gamma)

        time_step  = (theta - a*rj - lambda_*sigma)*delta_t
        stoch_step = sigma*nj*np.sqrt(delta_t)
        jump_step  = Jj*pj

        rj1 = rj + time_step + stoch_step + jump_step
        return rj1, nj, pj, Jj
