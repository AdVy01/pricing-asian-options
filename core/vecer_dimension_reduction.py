import numpy as np
from scipy.linalg import solve_banded

def vecer_fixed_asian_call(S0, K, T, r, sigma, M, N, theta=0.5):
    """
    Prices a Fixed-Strike Asian Call using the Vecer dimension reduction.
    q(t) = 1 - t/T
    """
    # Setup Grid
    dt = T / N
    z_min, z_max = -1.0, 1.0
    dz = (z_max - z_min) / M
    z = np.linspace(z_min, z_max, M + 1)
    nu = (dz**2) / dt
    
    # Initial Condition: u(T, z) = (z)^+
    u = np.maximum(z, 0)
    
    # Backward Induction
    for j in range(N - 1, -1, -1):
        t_curr = j * dt
        q_j = 1 - (t_curr / T)
        
        # Grid indices for inner points i = 1 to M-1
        zi = z[1:M]
        
        # Let A = sigma^2 * (q_j - z_i)^2 and B = dz * r * (q_j - z_i)
        A = (sigma**2) * (q_j - zi)**2
        B = dz * r * (q_j - zi)
        
        # LHS (Time j) coefficients
        L_left  = theta * (A - B)
        L_mid   = -2 * (theta * A + nu)
        L_right = theta * (A + B)
        
        # RHS (Time j+1) coefficients
        R_left  = -(1 - theta) * (A - B)
        R_mid   = 2 * ((1 - theta) * A - nu)
        R_right = -(1 - theta) * (A + B)
        
        # Construct RHS vector b
        # rhs_i = R_left*u_{i-1} + R_mid*u_i + R_right*u_{i+1}
        rhs = R_left * u[0:M-1] + R_mid * u[1:M] + R_right * u[2:M+1]
        
        # Boundary Conditions
        # Dirichlet: u[0,j] = 0.
        
        # Linear Extrapolation: u[M,j] = 2*u[M-1,j] - u[M-2,j]
        L_mid[-1]  += 2 * L_right[-1]
        L_left[-1] -= L_right[-1]
        
        # Solve Tridiagonal System
        # Format for solve_banded: [upper, diag, lower]
        ab = np.zeros((3, M - 1))
        ab[0, 1:] = L_right[:-1] # Upper diagonal
        ab[1, :]  = L_mid        # Main diagonal
        ab[2, :-1] = L_left[1:]  # Lower diagonal
        
        u_inner = solve_banded((1, 1), ab, rhs)
        
        # Update u-vector for the next time step (j-1)
        u[1:M] = u_inner
        u[0] = 0 
        u[M] = 2 * u[M-1] - u[M-2]
        
    # Vecer normalization: Price = S0 * u(0, z_0)
    # where z_0 = 1 - (K / S0)
    z_0 = 1 - (K / S0)
    price = S0 * np.interp(z_0, z, u)
    
    return max(price, 0)
