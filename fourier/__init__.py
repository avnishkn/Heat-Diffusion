"""
Fourier series methods for heat diffusion
"""

from .fourier_1d import Fourier1D, solve_heat_1d
from .fft_2d import FFT2D, solve_heat_2d

__all__ = ['Fourier1D', 'solve_heat_1d', 'FFT2D', 'solve_heat_2d']