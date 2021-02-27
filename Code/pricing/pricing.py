"""Module for pricing bonds"""

from copy import copy
import numpy as np
from evaluators.monte_carlo import MonteCarlo


class Pricing:
    """Class for pricing bonds using MC"""
    def __init__(self, model):
        self.model = model

    def bond_price(self, m, r0, n, T):
        """
        Calculates the price of bonds using MC
        """
        mc = MonteCarlo(self.model)
        
        return mc._simulate_paths_anti(m, r0, n, T)[0]
    
    def swap_rate(self, m: int, r0: float, n: int, freq: int, T:list):
        """
        Calculates the price of i-rate swaps using MC
        m, int: time steps per year,
        r0, float: i-rate today
        n, int: number of iterations to run,
        freq, int: frequency of payments in a year
        T, list: Payment dates of coupon and principal (in years) 
        """
        sr = np.empty(n)
        for i in range(n):
            z = np.empty(len(T))
            j = 0
            for t in T:
                z[j] = self.bond_price(m, r0, n, t)
                j = j + 1
            sr[i] = freq*(1-z[len(T)-1])/np.sum(z)
        return np.mean(sr), np.std(sr)
