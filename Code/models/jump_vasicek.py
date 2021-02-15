"Classes for Jump Vasicek Model"

import numpy as np

class JumpVasicek:
    """Vasicek Model with jumps"""
    def __init__(self, model_params: dict):
        self.model_params = model_params
    def increment(self, rj: float, dt: float, nj=None, Pj=None, Jj=None, Jj_pos=None):
        """
        Calculates next interest rate step
        """
        if nj is None or Pj is None or Jj is None:
            nj = np.random.normal()
            Pj = np.random.poisson(self.model_params["h"]*dt)
            Jj = np.random.normal(self.model_params["mu"], self.model_params["gamma"])
        time_step = self.model_params["kappa"]*(self.model_params["mu_r"] - rj)*dt
        stoch_step = self.model_params["sigma"]*np.sqrt(dt)*nj
        jump_step = Jj*Pj
        return rj + time_step + stoch_step + jump_step, nj, Pj, Jj, Jj_pos