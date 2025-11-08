"""
2D FFT methods for solving the heat equation on square/torus domains.
"""

import numpy as np
from typing import Callable, Tuple, Union
from manifolds import Square, Torus


class FFT2D:
    """
    2D FFT solver for heat equation: ∂u/∂t = κ Δu
    Uses FFT to diagonalize the Laplacian in Fourier space.
    """
    
    def __init__(self, domain: Union[Square, Torus], kappa: float = 1.0):
        """
        Parameters
        ----------
        domain : Square or Torus
            Domain for the heat equation
        kappa : float
            Diffusion coefficient (can be spatially varying)
        """
        self.domain = domain
        self.kappa = kappa
        
        # For heterogeneous media, kappa can be a function or array
        if callable(kappa):
            self.kappa_func = kappa
            self.kappa_const = None
        elif isinstance(kappa, (int, float)):
            self.kappa_const = float(kappa)
            self.kappa_func = None
        else:
            raise ValueError("kappa must be a constant or callable function")
    
    def compute_wavenumbers(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute wavenumber arrays for FFT.
        
        Parameters
        ----------
        N : int
            Number of grid points per dimension
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (kx, ky) wavenumber arrays
        """
        if isinstance(self.domain, Torus):
            kx, ky = self.domain.fourier_modes(N)
        else:
            # For square with periodic BCs, use FFT frequencies
            kx = np.fft.fftfreq(N, self.domain.L / N) * 2 * np.pi
            ky = np.fft.fftfreq(N, self.domain.L / N) * 2 * np.pi
        
        return kx, ky
    
    def laplacian_eigenvalues_fft(self, N: int) -> np.ndarray:
        """
        Compute Laplacian eigenvalues in Fourier space.
        For 2D: λ = -(kx² + ky²)
        
        Parameters
        ----------
        N : int
            Number of grid points per dimension
            
        Returns
        -------
        np.ndarray
            2D array of eigenvalues (shape: [N, N])
        """
        kx, ky = self.compute_wavenumbers(N)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        
        # Laplacian eigenvalue: -(kx² + ky²)
        eigenvalues = -(KX**2 + KY**2)
        return eigenvalues
    
    def decompose(self, u0: np.ndarray) -> np.ndarray:
        """
        Decompose initial condition using 2D FFT.
        
        Parameters
        ----------
        u0 : np.ndarray
            Initial condition (2D array, shape: [N, N])
            
        Returns
        -------
        np.ndarray
            Fourier coefficients (2D array in Fourier space)
        """
        # 2D FFT
        u_hat = np.fft.fft2(u0)
        return u_hat
    
    def evolve(self, u_hat: np.ndarray, t: float, dt: float = None) -> np.ndarray:
        """
        Evolve Fourier coefficients forward in time.
        
        For homogeneous media: u_hat(t) = u_hat(0) * exp(κ * λ * t)
        For heterogeneous media: use time-stepping
        
        Parameters
        ----------
        u_hat : np.ndarray
            Fourier coefficients at initial time
        t : float
            Time to evolve
        dt : float, optional
            Time step for heterogeneous media (if None, uses single step)
            
        Returns
        -------
        np.ndarray
            Evolved Fourier coefficients
        """
        N = u_hat.shape[0]
        eigenvalues = self.laplacian_eigenvalues_fft(N)
        
        if self.kappa_const is not None:
            # Homogeneous media: exact solution
            decay = np.exp(self.kappa_const * eigenvalues * t)
            return u_hat * decay
        else:
            # Heterogeneous media: need time-stepping
            # This is a simplified version - full implementation would handle
            # convolution in Fourier space
            if dt is None:
                dt = t / 100  # Default to 100 steps
            
            u_hat_t = u_hat.copy()
            n_steps = int(t / dt)
            dt_actual = t / n_steps
            
            for _ in range(n_steps):
                # Simplified: use average kappa (full version would be more complex)
                # For now, approximate with constant kappa
                if self.kappa_func is not None:
                    # Evaluate kappa at grid points (simplified)
                    x, y = self.domain.grid(N)
                    X, Y = np.meshgrid(x, y, indexing='ij')
                    kappa_vals = self.kappa_func(X, Y)
                    kappa_avg = np.mean(kappa_vals)
                else:
                    kappa_avg = 1.0
                
                decay = np.exp(kappa_avg * eigenvalues * dt_actual)
                u_hat_t = u_hat_t * decay
            
            return u_hat_t
    
    def reconstruct(self, u_hat: np.ndarray) -> np.ndarray:
        """
        Reconstruct solution from Fourier coefficients using inverse FFT.
        
        Parameters
        ----------
        u_hat : np.ndarray
            Fourier coefficients
            
        Returns
        -------
        np.ndarray
            Reconstructed solution (2D array)
        """
        u = np.fft.ifft2(u_hat)
        return np.real(u)


def solve_heat_2d(
    u0: Union[Callable, np.ndarray],
    domain: Union[Square, Torus],
    kappa: float = 1.0,
    t: float = 0.1,
    N: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D heat equation using FFT.
    
    Parameters
    ----------
    u0 : Callable or np.ndarray
        Initial condition function u0(x, y) or 2D array
    domain : Square or Torus
        Domain for the heat equation
    kappa : float
        Diffusion coefficient
    t : float
        Final time
    N : int
        Number of grid points per dimension
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (x, y, u) - spatial grids and solution at time t
    """
    # Create grid
    x, y = domain.grid(N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Evaluate initial condition
    if callable(u0):
        u0_vals = u0(X, Y)
    else:
        u0_vals = u0
    
    # Initialize solver
    solver = FFT2D(domain, kappa)
    
    # Decompose initial condition
    u_hat = solver.decompose(u0_vals)
    
    # Evolve in time
    u_hat_t = solver.evolve(u_hat, t)
    
    # Reconstruct solution
    u = solver.reconstruct(u_hat_t)
    
    return x, y, u