"""
Engine Package - Core Numerical Methods Implementations

Provides abstract base solver and concrete implementations of:
- Bisection Method
- Secant Method
- Gauss-Seidel Method
"""

from .base_solver import BaseSolver, SolverResult, SolverStep
from .bisection import BisectionMethod
from .secant import SecantMethod
from .gauss_seidel import GaussSeidelMethod

__all__ = [
    'BaseSolver',
    'SolverResult',
    'SolverStep',
    'BisectionMethod',
    'SecantMethod',
    'GaussSeidelMethod'
]
