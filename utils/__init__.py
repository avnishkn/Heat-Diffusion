"""
Utility functions for validation and analysis
"""

from .validation import compute_l2_error, convergence_study
from .energy import compute_energy, energy_decay

__all__ = ['compute_l2_error', 'convergence_study', 'compute_energy', 'energy_decay']