"""
2-sphere domain for heat diffusion using spherical harmonics.
"""

import numpy as np
from typing import Tuple
from scipy.special import sph_harm


class Sphere2D:
    """
    Two-dimensional sphere S² for heat diffusion.
    Uses spherical harmonics Y_ℓ^m as eigenfunctions of the Laplacian.
    """
    
    def __init__(self, radius: float = 1.0):
        """
        Parameters
        ----------
        radius : float
            Radius of the sphere
        """
        self.radius = radius
    
    def eigenfunctions(self, ell: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Compute (ℓ,m)-th spherical harmonic Y_ℓ^m(θ, φ).
        
        Parameters
        ----------
        ell : int
            Degree (ℓ >= 0)
        m : int
            Order (-ℓ <= m <= ℓ)
        theta : np.ndarray
            Colatitude (polar angle) in [0, π]
        phi : np.ndarray
            Azimuthal angle in [0, 2π]
            
        Returns
        -------
        np.ndarray
            Spherical harmonic values (complex)
        """
        if abs(m) > ell:
            raise ValueError(f"|m| must be <= ℓ. Got m={m}, ℓ={ell}")
        
        # scipy.sph_harm uses (m, ell) order and expects angles in radians
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        return sph_harm(m, ell, PHI, THETA)
    
    def eigenvalues(self, ell: int) -> float:
        """
        Compute ℓ-th eigenvalue of the Laplacian on the sphere.
        
        The eigenvalue depends only on ℓ, not m.
        λ_ℓ = -ℓ(ℓ+1) / R²
        
        Parameters
        ----------
        ell : int
            Degree (ℓ >= 0)
            
        Returns
        -------
        float
            Eigenvalue λ_ℓ
        """
        return -ell * (ell + 1) / (self.radius ** 2)
    
    def grid(self, N_theta: int, N_phi: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create spatial grid for the sphere.
        
        Parameters
        ----------
        N_theta : int
            Number of points in colatitude (θ) direction
        N_phi : int
            Number of points in azimuthal (φ) direction
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (theta, phi) grid points
        """
        # Colatitude: 0 to π
        theta = np.linspace(0, np.pi, N_theta)
        # Azimuthal: 0 to 2π (periodic)
        phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
        return theta, phi
    
    def to_cartesian(self, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert spherical coordinates to Cartesian.
        
        Parameters
        ----------
        theta : np.ndarray
            Colatitude
        phi : np.ndarray
            Azimuthal angle
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (x, y, z) Cartesian coordinates
        """
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        x = self.radius * np.sin(THETA) * np.cos(PHI)
        y = self.radius * np.sin(THETA) * np.sin(PHI)
        z = self.radius * np.cos(THETA)
        return x, y, z
    
    def normalize_spherical_harmonic(self, ell: int, m: int) -> float:
        """
        Normalization constant for spherical harmonic Y_ℓ^m.
        scipy.sph_harm already includes normalization, but this is useful for reference.
        
        Parameters
        ----------
        ell : int
            Degree
        m : int
            Order
            
        Returns
        -------
        float
            Normalization constant (scipy already normalizes, so this returns 1.0)
        """
        # scipy.sph_harm uses the physics convention with normalization
        # ∫ |Y_ℓ^m|² dΩ = 1
        return 1.0