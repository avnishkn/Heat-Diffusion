"""
2D square domain for heat diffusion.
"""

import numpy as np
from typing import Tuple, Literal


class Square:
    """
    Two-dimensional square domain [0, L] × [0, L] with various boundary conditions.
    """
    
    def __init__(self, L: float = 1.0, bc_type: Literal['dirichlet', 'neumann', 'periodic'] = 'dirichlet'):
        """
        Parameters
        ----------
        L : float
            Side length of the square
        bc_type : str
            Boundary condition type: 'dirichlet', 'neumann', or 'periodic'
        """
        self.L = L
        self.bc_type = bc_type
    
    def eigenfunctions(self, n: int, m: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute (n,m)-th eigenfunction of the Laplacian on the square.
        
        Parameters
        ----------
        n : int
            Mode number in x-direction (n >= 1)
        m : int
            Mode number in y-direction (m >= 1)
        x : np.ndarray
            x-coordinates
        y : np.ndarray
            y-coordinates
            
        Returns
        -------
        np.ndarray
            Eigenfunction values (2D array matching x, y meshgrid)
        """
        if self.bc_type == 'dirichlet':
            # u = 0 on boundaries: sin(nπx/L) sin(mπy/L)
            X, Y = np.meshgrid(x, y, indexing='ij')
            return (2 / self.L) * np.sin(n * np.pi * X / self.L) * np.sin(m * np.pi * Y / self.L)
        elif self.bc_type == 'neumann':
            # u' = 0 on boundaries: cos(nπx/L) cos(mπy/L)
            X, Y = np.meshgrid(x, y, indexing='ij')
            if n == 0 and m == 0:
                return np.ones_like(X) / self.L
            elif n == 0:
                return np.sqrt(2) / self.L * np.cos(m * np.pi * Y / self.L)
            elif m == 0:
                return np.sqrt(2) / self.L * np.cos(n * np.pi * X / self.L)
            else:
                return (2 / self.L) * np.cos(n * np.pi * X / self.L) * np.cos(m * np.pi * Y / self.L)
        elif self.bc_type == 'periodic':
            # Periodic: exp(i2πnx/L) exp(i2πmy/L)
            X, Y = np.meshgrid(x, y, indexing='ij')
            return np.exp(1j * 2 * np.pi * n * X / self.L) * np.exp(1j * 2 * np.pi * m * Y / self.L) / self.L
        else:
            raise ValueError(f"Unknown boundary condition: {self.bc_type}")
    
    def eigenvalues(self, n: int, m: int) -> float:
        """
        Compute (n,m)-th eigenvalue of the Laplacian.
        
        Parameters
        ----------
        n : int
            Mode number in x-direction
        m : int
            Mode number in y-direction
            
        Returns
        -------
        float
            Eigenvalue λ_{n,m} = -π²(n² + m²)/L²
        """
        if self.bc_type == 'dirichlet' or self.bc_type == 'neumann':
            return -np.pi**2 * (n**2 + m**2) / (self.L**2)
        elif self.bc_type == 'periodic':
            return -4 * np.pi**2 * (n**2 + m**2) / (self.L**2)
        else:
            raise ValueError(f"Unknown boundary condition: {self.bc_type}")
    
    def grid(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create spatial grid for the square.
        
        Parameters
        ----------
        N : int
            Number of grid points per dimension
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (x, y) grid points
        """
        if self.bc_type == 'dirichlet':
            # For Dirichlet: exclude boundaries
            x = np.linspace(0, self.L, N + 2)[1:-1]
            y = np.linspace(0, self.L, N + 2)[1:-1]
        else:
            # For Neumann/Periodic: include boundaries (periodic uses endpoint=False)
            if self.bc_type == 'periodic':
                x = np.linspace(0, self.L, N, endpoint=False)
                y = np.linspace(0, self.L, N, endpoint=False)
            else:
                x = np.linspace(0, self.L, N)
                y = np.linspace(0, self.L, N)
        
        return x, y