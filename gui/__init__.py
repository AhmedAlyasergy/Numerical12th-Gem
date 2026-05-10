"""
GUI Module - Main Application Interface

Provides a modern PyQt6-based dark-mode dashboard for the Numerical Methods Workbench.
"""

from .main_window import MainWindow
from .themes import DarkTheme

__all__ = ["MainWindow", "DarkTheme"]
