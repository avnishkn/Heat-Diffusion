"""
PDE numerical solvers for the heat equation
"""

from .finite_difference import FiniteDifference1D, FiniteDifference2D
from .spectral_methods import SpectralSolver

__all__ = ['FiniteDifference1D', 'FiniteDifference2D', 'SpectralSolver']