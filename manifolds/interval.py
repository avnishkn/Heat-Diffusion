"""
1D interval domain [0, L] for heat diffusion.
"""

import numpy as np
from typing import Tuple, Literal


class Interval:
    """
    One-dimensional interval domain [0, L] with various boundary conditions.
    """
    
    def __init__(self, L: float = 1.0, bc_type: Literal['dirichlet', 'neumann', 'periodic'] = 'dirichlet'):
        """
        Parameters
        ----------
        L : float
            Length of the interval
        bc_type : str
            Boundary condition type: 'dirichlet' (ends fixed at 0 temp), 'neumann' (end gradients fixed at 0), or 'periodic' (ends connected)
        """
        self.L = L
        self.bc_type = bc_type
    
    def eigenfunctions(self, n: int, x: np.ndarray) -> np.ndarray:
        """
        Compute n-th eigenfunction of the Laplacian on the interval.
        
        Parameters
        ----------
        n : int
            Mode number (n >= 1)
        x : np.ndarray
            Spatial points in [0, L]
            
        Returns
        -------
        np.ndarray
            Eigenfunction values
        """
        if self.bc_type == 'dirichlet':
            # u(0) = u(L) = 0: sin(nπx/L)
            return np.sqrt(2 / self.L) * np.sin(n * np.pi * x / self.L)
        elif self.bc_type == 'neumann':
            # u'(0) = u'(L) = 0: cos(nπx/L)
            if n == 0:
                return np.ones_like(x) / np.sqrt(self.L)
            return np.sqrt(2 / self.L) * np.cos(n * np.pi * x / self.L)
        elif self.bc_type == 'periodic':
            # Periodic: exp(i2πnx/L)
            return np.exp(1j * 2 * np.pi * n * x / self.L) / np.sqrt(self.L)
        else:
            raise ValueError(f"Unknown boundary condition: {self.bc_type}")
    
    def eigenvalues(self, n: int) -> float:
        """
        Compute n-th eigenvalue of the Laplacian.
        
        Parameters
        ----------
        n : int
            Mode number
            
        Returns
        -------
        float
            Eigenvalue λ_n = -(nπ/L)² for Dirichlet/Neumann
        """
        if self.bc_type == 'dirichlet' or self.bc_type == 'neumann':
            return -(n * np.pi / self.L) ** 2
        elif self.bc_type == 'periodic':
            return -(2 * np.pi * n / self.L) ** 2
        else:
            raise ValueError(f"Unknown boundary condition: {self.bc_type}")
    
    def grid(self, N: int) -> np.ndarray:
        """
        Create spatial grid for the interval.
        
        Parameters
        ----------
        N : int
            Number of grid points
            
        Returns
        -------
        np.ndarray
            Grid points
        """
        if self.bc_type == 'dirichlet':
            # For Dirichlet: exclude boundaries (they're zero)
            return np.linspace(0, self.L, N + 2)[1:-1]
        else:
            # For Neumann/Periodic: include boundaries
            return np.linspace(0, self.L, N, endpoint=False)