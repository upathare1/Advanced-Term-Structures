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
        
        return mc._simulate_paths(m, r0, n, T)[0]
    
    def swap_rate(self, m, r0, n, freq, T:list):
        """
        Calculates the price of bonds using MC
        m, int: time steps per year,
        r0, float: i-rate today
        n, 1x(T*m) np.array, array of draws,
        freq : frequency of payments in a year
        T, list: Payment dates of coupon and principal (in years) 
        """
        mc = MonteCarlo(self.model)
        z = np.empty(len(T))
        i = 0
        for t in T:
            z[i] = mc._simulate_paths(m, r0, n, t)[0]
            i = i + 1
        return freq*(1-z[len(T)-1])/np.sum(z)
    

