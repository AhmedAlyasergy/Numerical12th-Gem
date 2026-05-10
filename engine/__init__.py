"""
Numerical Methods Engine Module

Provides implementations of various numerical methods for root finding
and system solving with detailed step-by-step execution tracking.
"""

from .bisection import BisectionMethod
from .secant import SecantMethod
from .gauss_seidel import GaussSeidelMethod

__all__ = ["BisectionMethod", "SecantMethod", "GaussSeidelMethod"]
