"""Classes for Spot Rate models and Monte Carlo simulations"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        stoch_step = sigma*np.random.normal()*np.sqrt(delta_t)
        jump_step  = np.random.normal(mu, gamma)*np.random.poisson(h*delta_t)
        
        rj1 = rj + time_step + stoch_step + jump_step
    #     rj1 = rj + time_step + stoch_step
        return rj1, nj, pj, Jj

class MonteCarlo:
    """Class for running MC experiments"""
    def __init__(self, spot_rate_model, r, t, T, m, n):
        self.spot_rate_model = spot_rate_model
        self.t = t
        self.T = T
        self.m = m
        self.n = n
        self.r = r

    def _simulate_path(self):
        """Simulate a single path"""
        r = [self.r]
        n = []
        p = []
        J = []
        delta_t = (self.T-self.t)/self.m
        for j in range(self.m):
            rj1, nj, pj, Jj = self.spot_rate_model.increment(r[j], self.t, self.T, self.m)
            r.append(rj1)
            n.append(nj)
            p.append(pj)
            J.append(Jj)
        prices = np.exp(-np.cumsum(np.multiply(r[1:], delta_t)))
        prices = np.insert(prices, 0, np.exp(-self.t*self.r))
        return prices, r, n, p, J
    
    def _simulate_antithetical_path(self, n, p, J):
        """Simulate antithetical path"""
        n_anti = -n
        J_anti = self.spot_rate_model.model_params["mu"] - (J-self.spot_rate_model.model_params["mu"])
        p_anti = p
        r = []
        delta_t = (self.T-self.t)/self.m
        for j in range(self.m):
            rj1, _, __, ___ = self.spot_rate_model.increment(r[j], self.t, self.T, self.m, n_anti[j], p_anti[j], J_anti[j])
            r.append(rj1)
        prices = np.exp(-np.cumsum(np.multiply(r[1:], delta_t)))
        prices = np.insert(prices, 0, np.exp(-self.t*self.r))
        return prices, r, n_anti, p_anti, J_anti
    
    def _simulate_price(self):
        """Classical MC simulation"""
        prices = np.empty((self.n, self.m+1), dtype=np.float)
        rs = np.empty((self.n, self.m+1), dtype=np.float)
        for i in range(self.n):
            prices[i, :], rs[i, :], _, __, ___ = self._simulate_path()
        return np.mean(prices, axis=0), rs
    
    def _simulate_price_antithetical(self):
        """Antithetical Variates MC"""
        prices = np.empty((self.n, self.m+1), dtype=np.float)
        rs = np.empty((2*self.n, self.m+1), dtype=np.float)
        for i in range(self.n):
            prices[2*i, :], rs[2*i, :], n, p, j = self._simulate_path()
            prices[2*i + 1, :], rs[2*i + 1, :], _, __, ___ = self._simulate_antithetical_path(n, p, j)
        return np.mean(prices, axis=0), rs

    def plot_price_curve(self, type_="classical"):
        """Plot price curve"""
        if type_ == "classical":
            Prices, rs = self._simulate_price_antithetical()
        elif type_ == "antithetic":
            Prices, rs = self._simulate_price()
        pd.Series(Prices, index=np.linspace(self.t, self.T, self.m+1)).plot()
        pd.DataFrame(rs.T, index=np.linspace(self.t, self.T, self.m+1)).plot(legend=False)
        return Prices
