"""
2D torus domain (periodic square) for heat diffusion.
"""

import numpy as np
from typing import Tuple


class Torus:
    """
    Two-dimensional torus domain (periodic square) [0, L] × [0, L] with periodic boundary conditions.
    Topologically equivalent to T² = S¹ × S¹.
    """
    
    def __init__(self, L: float = 1.0):
        """
        Parameters
        ----------
        L : float
            Side length of the periodic square (period in both directions)
        """
        self.L = L
    
    def eigenfunctions(self, n: int, m: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute (n,m)-th eigenfunction of the Laplacian on the torus.
        
        For periodic BCs: exp(i2πnx/L) exp(i2πmy/L)
        
        Parameters
        ----------
        n : int
            Mode number in x-direction (can be negative)
        m : int
            Mode number in y-direction (can be negative)
        x : np.ndarray
            x-coordinates
        y : np.ndarray
            y-coordinates
            
        Returns
        -------
        np.ndarray
            Eigenfunction values (2D array matching x, y meshgrid)
        """
        X, Y = np.meshgrid(x, y, indexing='ij')
        return np.exp(1j * 2 * np.pi * n * X / self.L) * np.exp(1j * 2 * np.pi * m * Y / self.L) / self.L
    
    def eigenvalues(self, n: int, m: int) -> float:
        """
        Compute (n,m)-th eigenvalue of the Laplacian on the torus.
        
        Parameters
        ----------
        n : int
            Mode number in x-direction
        m : int
            Mode number in y-direction
            
        Returns
        -------
        float
            Eigenvalue λ_{n,m} = -4π²(n² + m²)/L²
        """
        return -4 * np.pi**2 * (n**2 + m**2) / (self.L**2)
    
    def grid(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create spatial grid for the torus (periodic).
        
        Parameters
        ----------
        N : int
            Number of grid points per dimension
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (x, y) grid points (periodic, endpoint=False)
        """
        x = np.linspace(0, self.L, N, endpoint=False)
        y = np.linspace(0, self.L, N, endpoint=False)
        return x, y
    
    def fourier_modes(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Fourier mode indices for FFT.
        
        Parameters
        ----------
        N : int
            Number of grid points per dimension
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (kx, ky) wavenumber arrays for FFT
        """
        # FFT frequencies: [0, 1, 2, ..., N/2-1, -N/2, ..., -1] * 2π/L
        kx = np.fft.fftfreq(N, self.L / N) * 2 * np.pi
        ky = np.fft.fftfreq(N, self.L / N) * 2 * np.pi
        return kx, ky