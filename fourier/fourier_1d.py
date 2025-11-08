"""
1D Fourier series methods for solving the heat equation on an interval.
"""

import numpy as np
from typing import Callable, Tuple
from manifolds import Interval


class Fourier1D:
    """
    Fourier series solver for 1D heat equation: ∂u/∂t = κ Δu
    """
    
    def __init__(self, interval: Interval, kappa: float = 1.0, N_modes: int = 50):
        """
        Parameters
        ----------
        interval : Interval
            Interval domain
        kappa : float
            Diffusion coefficient
        N_modes : int
            Number of Fourier modes to use
        """
        self.interval = interval
        self.kappa = kappa
        self.N_modes = N_modes
        
        # Precompute eigenvalues for all modes
        self.eigenvalues = np.array([interval.eigenvalues(n) for n in range(1, N_modes + 1)])
    
    def decompose(self, u0: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Decompose initial condition u0 into Fourier coefficients.
        
        Parameters
        ----------
        u0 : np.ndarray
            Initial condition values at grid points x
        x : np.ndarray
            Spatial grid points
            
        Returns
        -------
        np.ndarray
            Fourier coefficients (amplitudes for each mode)
        """
        N = len(x)
        coeffs = np.zeros(self.N_modes, dtype=complex)
        
        for n in range(1, self.N_modes + 1):
            # Compute inner product: <u0, φ_n>
            phi_n = self.interval.eigenfunctions(n, x)
            if self.interval.bc_type == 'periodic':
                # For periodic, use complex inner product
                coeffs[n-1] = np.trapz(u0 * np.conj(phi_n), x)
            else:
                # For Dirichlet/Neumann, eigenfunctions are real
                coeffs[n-1] = np.trapz(u0 * phi_n, x)
        
        return coeffs
    
    def evolve(self, coeffs: np.ndarray, t: float) -> np.ndarray:
        """
        Evolve Fourier coefficients forward in time.
        
        Each mode decays as: c_n(t) = c_n(0) * exp(κ * λ_n * t)
        
        Parameters
        ----------
        coeffs : np.ndarray
            Initial Fourier coefficients
        t : float
            Time
        
        Returns
        -------
        np.ndarray
            Evolved Fourier coefficients
        """
        # Exponential decay: exp(κ * λ_n * t)
        decay = np.exp(self.kappa * self.eigenvalues * t)
        return coeffs * decay
    
    def reconstruct(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Reconstruct solution from Fourier coefficients.
        
        Parameters
        ----------
        coeffs : np.ndarray
            Fourier coefficients
        x : np.ndarray
            Spatial grid points
            
        Returns
        -------
        np.ndarray
            Reconstructed solution u(x, t)
        """
        u = np.zeros_like(x, dtype=complex)
        
        for n in range(1, self.N_modes + 1):
            phi_n = self.interval.eigenfunctions(n, x)
            u += coeffs[n-1] * phi_n
        
        # Return real part (imaginary part should be ~0 for real initial conditions)
        return np.real(u)


def solve_heat_1d(
    u0: Callable[[np.ndarray], np.ndarray],
    interval: Interval,
    kappa: float = 1.0,
    t: float = 0.1,
    N: int = 100,
    N_modes: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve 1D heat equation using Fourier series.
    
    Parameters
    ----------
    u0 : Callable
        Initial condition function u0(x)
    interval : Interval
        Interval domain
    kappa : float
        Diffusion coefficient
    t : float
        Final time
    N : int
        Number of spatial grid points
    N_modes : int
        Number of Fourier modes
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (x, u) - spatial grid and solution at time t
    """
    # Create grid
    x = interval.grid(N)
    
    # Evaluate initial condition
    u0_vals = u0(x)
    
    # Initialize solver
    solver = Fourier1D(interval, kappa, N_modes)
    
    # Decompose initial condition
    coeffs = solver.decompose(u0_vals, x)
    
    # Evolve in time
    coeffs_t = solver.evolve(coeffs, t)
    
    # Reconstruct solution
    u = solver.reconstruct(coeffs_t, x)
    
    return x, u