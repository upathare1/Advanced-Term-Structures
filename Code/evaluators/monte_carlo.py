"Classes for running Monte Carlo Simulations"

import numpy as np
import pandas as pd
from IPython.display import display

class MonteCarlo:
    """Class for running MC experiments"""
    def __init__(self, spot_rate_model, r, t, T, m, n):
        self.spot_rate_model = spot_rate_model
        self.t = t
        self.T = T
        self.m = m
        self.n = n
        self.r = r

    def _is_integer(self, fl: float):
        """Helper function to check if float is an integer"""
        return fl.is_integer()

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
        n_anti = np.negative(n)
        J_anti = np.subtract(2*self.spot_rate_model.model_params["mu"], J)
        p_anti = p
        r = [self.r]
        delta_t = (self.T-self.t)/self.m
        for j in range(self.m):
            rj1, _, __, ___ = self.spot_rate_model.increment(r[j], self.t, self.T, self.m, n_anti[j], p_anti[j], J_anti[j])
            r.append(rj1)
        prices = np.exp(-np.cumsum(np.multiply(r[1:], delta_t)))
        prices = np.insert(prices, 0, np.exp(-self.t*self.r))
        return prices, r
    
    def _simulate_price(self):
        """Classical MC simulation"""
        prices = np.empty((self.n, self.m+1), dtype=np.float)
        rs = np.empty((self.n, self.m+1), dtype=np.float)
        for i in range(self.n):
            prices[i, :], rs[i, :], _, __, ___ = self._simulate_path()
        return prices, rs
    
    def _simulate_price_antithetical(self):
        """Antithetical Variates MC"""
        prices = np.empty((2*self.n, self.m+1), dtype=np.float)
        rs = np.empty((2*self.n, self.m+1), dtype=np.float)
        for i in range(self.n):
            prices[2*i, :], rs[2*i, :], n, p, j = self._simulate_path()
            prices[2*i + 1, :], rs[2*i + 1, :] = self._simulate_antithetical_path(n, p, j)
        return prices, rs

    def plot_price_curve(self, type_="classical"):
        """Plot price curve"""
        if type_ == "antithetic":
            prices, rs = self._simulate_price_antithetical()
        elif type_ == "classical":
            prices, rs = self._simulate_price()
        mean_prices = np.mean(prices, axis=0)
        sd_prices = np.std(prices, axis=0)
        time_index = pd.Index(np.linspace(self.t, self.T, self.m+1))
        time_index.name = "Year"
        mean_prices = pd.Series(mean_prices, index=time_index)
        sd_prices = pd.Series(sd_prices, index=time_index)
        i_rates = pd.DataFrame(rs.T, index=time_index)
        mean_prices.plot()
        i_rates.plot(legend=False)
        return prices, mean_prices, sd_prices, i_rates
