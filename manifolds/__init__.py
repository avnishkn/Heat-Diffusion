"""
Geometric domains for heat diffusion
"""

from .interval import Interval
from .square import Square
from .torus import Torus
from .sphere import Sphere2D

__all__ = ['Interval', 'Square', 'Torus', 'Sphere2D']