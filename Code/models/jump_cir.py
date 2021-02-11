"Classes for Jump CIR Model"

import numpy as np
from scipy.stats import truncnorm
import math

class JumpCIR:
    """CIR Model with jumps"""
    def __init__(self, model_params: dict):
        self.model_params = model_params
    def increment(self, rj: float, dt: float, nj=None, Pj=None, Jj=None, Jj_pos=None):
        """
        Calculates next interest rate step
        """
        if nj is None or Pj is None: 
            nj = np.random.normal()
            Pj = np.random.poisson(self.model_params["h"]*dt)
        time_step = self.model_params["kappa_r"]*(self.model_params["mu_r"] - rj)*dt
        stoch_step = self.model_params["sigma"]*np.sqrt(rj)*np.sqrt(dt)*nj
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
        """
        Calculates truncated jump for CIR
        """
        if Pj == 0:
            Jj = 0
        else:
            lower_bound = (-rj - time_step - stoch_step)/Pj
            upper_bound = -lower_bound
            Jj = truncnorm(lower_bound, upper_bound, loc=self.model_params["mu"], scale=self.model_params["gamma"]).rvs(1)[0]
        return Jj