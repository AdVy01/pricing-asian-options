import numpy as np
import math
from scipy.integrate import quad
from scipy.special import gamma
import matplotlib.pyplot as plt

from mc_control_variate import AsianOptionMC_MC

def C_hat(p, nu, alpha):
    """
    Represents the Laplace transform of the scaled price.
    """
    mu = np.sqrt(2 * p + nu**2)
    
    # Powers used in the integrand
    pow1 = 0.5 * (mu - nu) - 2
    pow2 = 0.5 * (mu + nu) + 1
    
    # The inner integral over 'z'
    def inner_integrand(z):
        return np.exp(-z) * (z**pow1) * ((1 - 2 * (-alpha) * z)**pow2)

    # We integrate real and imaginary parts separately for stability
    upper_limit = 1 / (2 * (-alpha))
    re_int, _ = quad(lambda x: np.real(inner_integrand(x)), 0, upper_limit, limit=100)
    im_int, _ = quad(lambda x: np.imag(inner_integrand(x)), 0, upper_limit, limit=100)
    res_integral = complex(re_int, im_int)
    
    denominator = p * (p - 2 - 2 * nu) * gamma(0.5 * (mu - nu) - 1)
    
    return res_integral / denominator

def euler_invert_laplace(F, t, N=15, M=11):
    """
    Evaluates the inverse Laplace transform using the Euler algorithm (Abate & Whitt, 1995).
    
    Parameters:
    F : callable
        The Laplace transform function F(p) taking a complex argument.
    t : float
        The time variable at which to evaluate the inverse (must be > 0).
    N : int
        Number of terms in the basic alternating series.
    M : int
        Number of terms for the Euler summation acceleration.
    """
    # A = 18.4 targets roughly 8 decimal places of accuracy.
    A = 18.4 
    
    # Array to store the terms of the alternating series
    su = np.zeros(N + M + 1)
    
    # First term (k = 0)
    su[0] = 0.5 * np.real(F(A / (2 * t)))
    
    # Remaining terms (k = 1 to N + M)
    for k in range(1, N + M + 1):
        p = (A + 2 * k * np.pi * 1j) / (2 * t)
        term = np.real(F(p))
        if k % 2 != 0:
            su[k] = -term
        else:
            su[k] = term
            
    # Calculate partial sums of the series
    partial_sums = np.cumsum(su)
    
    # Apply Euler summation acceleration
    def choose(n, k):
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
        
    res = 0.0
    for k in range(M + 1):
        weight = choose(M, k) / (2 ** M)
        res += weight * partial_sums[N + k]
        
    return (np.exp(A / 2) / t) * res

def calculate_asian_call_price(r, sigma, T, t0, t, A, S, K, N=15, M=15):
    nu = (2 * r / (sigma**2)) - 1
    tau_2 = (sigma**2 / 4) * (T - t)
    alpha = -(sigma**2 / (4 * S)) * (K * (T - t0) - (t - t0) * A)
    
    def F_transform(p):
        return C_hat(p, nu, alpha)

    # tau_2 is the 'time' variable in the Geman-Yor model
    integral_val = euler_invert_laplace(F_transform, tau_2, N, M)

    # Final scaling to get the actual option price
    scaling = 4 * S * np.exp(-r * (T - t)) / (sigma**2 * (T - t0))
    option_price = integral_val * scaling
    return option_price
