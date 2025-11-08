"""
Spherical harmonics based methods for solving the heat equation on the sphere.
"""

import numpy as np
from typing import Callable, Tuple
from manifolds import Sphere2D
from scipy.special import sph_harm


class SphericalHarmonicsSolver:
    """
    Solve the heat equation on the sphere using spherical harmonics.
    Keeps track of coefficients a_{ℓ,m}(t) for each mode.
    """
    
    def __init__(self, sphere: Sphere2D, ell_max: int = 10, kappa: float = 1.0):
        """
        Parameters
        ----------
        sphere : Sphere2D
            Sphere domain
        ell_max : int
            Maximum degree ℓ to include (m runs from -ℓ to ℓ)
        kappa : float
            Diffusion coefficient
        """
        self.sphere = sphere
        self.ell_max = ell_max
        self.kappa = kappa
        
        # Precompute eigenvalues λ_ℓ = -ℓ(ℓ+1)/R²
        self.eigenvalues = np.array([sphere.eigenvalues(ell) for ell in range(ell_max + 1)])
    
    def prepare_grid(self, N_theta: int, N_phi: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare spherical coordinate grids.
        
        Parameters
        ----------
        N_theta : int
            Number of grid points in theta (colatitude)
        N_phi : int
            Number of grid points in phi (azimuthal)
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (theta, phi) arrays
        """
        theta, phi = self.sphere.grid(N_theta, N_phi)
        return theta, phi
    
    def decompose(self, u0: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                  theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Decompose initial condition into spherical harmonic coefficients.
        
        Parameters
        ----------
        u0 : Callable
            Initial condition function u0(theta, phi)
        theta : np.ndarray
            Theta grid (1D array)
        phi : np.ndarray
            Phi grid (1D array)
            
        Returns
        -------
        np.ndarray
            Coefficients a_{ℓ,m} (complex array of shape [ℓ_max+1, 2ℓ_max+1])
        """
        N_theta, N_phi = len(theta), len(phi)
        dtheta = theta[1] - theta[0]
        dphi = phi[1] - phi[0]
        
        # Create meshgrid
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        # Evaluate initial condition
        u0_vals = u0(THETA, PHI)
        
        # Initialize coefficients a_{ℓ,m}
        coeffs = np.zeros((self.ell_max + 1, 2 * self.ell_max + 1), dtype=complex)
        
        # Integration measure: sin(θ) dθ dφ
        weight = np.sin(THETA)
        
        for ell in range(self.ell_max + 1):
            for m in range(-ell, ell + 1):
                Y_lm = sph_harm(m, ell, PHI, THETA)
                integrand = u0_vals * np.conj(Y_lm) * weight
                coeff = np.sum(integrand) * dtheta * dphi
                coeffs[ell, m + self.ell_max] = coeff
        
        return coeffs
    
    def evolve(self, coeffs: np.ndarray, t: float) -> np.ndarray:
        """
        Evolve spherical harmonic coefficients forward in time.
        a_{ℓ,m}(t) = a_{ℓ,m}(0) * exp(κ λ_ℓ t)
        
        Parameters
        ----------
        coeffs : np.ndarray
            Initial coefficients (complex array)
        t : float
            Time
            
        Returns
        -------
        np.ndarray
            Evolved coefficients
        """
        ell_indices = np.arange(self.ell_max + 1)
        decay = np.exp(self.kappa * self.eigenvalues[:, None] * t)
        coeffs_t = coeffs * decay
        return coeffs_t
    
    def reconstruct(self, coeffs: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Reconstruct function from spherical harmonic coefficients.
        
        Parameters
        ----------
        coeffs : np.ndarray
            Spherical harmonic coefficients
        theta : np.ndarray
            Theta grid
        phi : np.ndarray
            Phi grid
            
        Returns
        -------
        np.ndarray
            Reconstructed function u(θ, φ)
        """
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        u = np.zeros_like(THETA, dtype=complex)
        
        for ell in range(self.ell_max + 1):
            for m in range(-ell, ell + 1):
                Y_lm = sph_harm(m, ell, PHI, THETA)
                coeff = coeffs[ell, m + self.ell_max]
                u += coeff * Y_lm
        
        return np.real(u)


def solve_heat_sphere(
    u0: Callable[[np.ndarray, np.ndarray], np.ndarray],
    sphere: Sphere2D,
    kappa: float = 1.0,
    t: float = 0.1,
    N_theta: int = 50,
    N_phi: int = 100,
    ell_max: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve heat equation on the sphere using spherical harmonics.
    
    Parameters
    ----------
    u0 : Callable
        Initial condition function u0(theta, phi)
    sphere : Sphere2D
        Sphere domain
    kappa : float
        Diffusion coefficient
    t : float
        Final time
    N_theta : int
        Number of grid points in theta
    N_phi : int
        Number of grid points in phi
    ell_max : int
        Maximum degree ℓ to include
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (theta, phi, u) - grids and solution at time t
    """
    solver = SphericalHarmonicsSolver(sphere, ell_max, kappa)
    theta, phi = solver.prepare_grid(N_theta, N_phi)
    
    # Decompose initial condition
    coeffs = solver.decompose(u0, theta, phi)
    
    # Evolve coefficients
    coeffs_t = solver.evolve(coeffs, t)
    
    # Reconstruct solution
    u = solver.reconstruct(coeffs_t, theta, phi)
    
    return theta, phi, u