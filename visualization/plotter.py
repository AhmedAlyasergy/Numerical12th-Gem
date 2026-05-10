"""
Numerical Plotter

Provides real-time plotting of functions, roots, and convergence analysis.
"""

from typing import Callable, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class NumericalPlotter:
    """
    Plotter for visualizing numerical method results.
    
    Supports:
    - Function plots with root markers
    - Step-by-step iteration visualization
    - Convergence history (error vs iteration)
    - Multiple plot types (function, convergence, steps)
    """
    
    # Dark theme colors
    BG_COLOR = '#1e1e1e'
    GRID_COLOR = '#333333'
    AXIS_COLOR = '#cccccc'
    
    @staticmethod
    def create_function_plot(
        f: Callable[[float], float],
        a: float,
        b: float,
        root: float = None,
        iterations: List[float] = None,
        title: str = "Function Plot"
    ) -> Figure:
        """
        Create a plot of function with marked root and iterations.
        
        Args:
            f: Function to plot
            a: Left boundary
            b: Right boundary
            root: Root location to mark
            iterations: List of x-values from iterations
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=NumericalPlotter.BG_COLOR)
        
        # Generate function values
        x = np.linspace(a, b, 1000)
        try:
            y = np.array([f(xi) for xi in x])
        except Exception as e:
            ax.text(0.5, 0.5, f"Error evaluating function: {str(e)}",
                   ha='center', va='center', transform=ax.transAxes,
                   color='red', fontsize=12)
            return fig
        
        # Plot function
        ax.plot(x, y, 'b-', linewidth=2.5, label='f(x)', alpha=0.9)
        ax.axhline(y=0, color=NumericalPlotter.GRID_COLOR, linestyle='--', alpha=0.5)
        ax.axvline(x=0, color=NumericalPlotter.GRID_COLOR, linestyle='--', alpha=0.5)
        
        # Mark root
        if root is not None:
            try:
                ax.plot(root, f(root), 'r*', markersize=20, label=f'Root ≈ {root:.6f}', zorder=5)
            except:
                pass
        
        # Mark iteration points
        if iterations:
            try:
                iter_y = [f(xi) for xi in iterations]
                ax.plot(iterations, iter_y, 'go', markersize=6, alpha=0.7, 
                       label=f'Iterations (n={len(iterations)})')
                # Draw vertical lines to x-axis
                for xi in iterations:
                    ax.plot([xi, xi], [f(xi), 0], 'g--', alpha=0.3, linewidth=1)
            except:
                pass
        
        # Styling
        ax.set_facecolor(NumericalPlotter.BG_COLOR)
        ax.set_xlabel('x', color=NumericalPlotter.AXIS_COLOR, fontsize=11)
        ax.set_ylabel('f(x)', color=NumericalPlotter.AXIS_COLOR, fontsize=11)
        ax.set_title(title, color=NumericalPlotter.AXIS_COLOR, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2, color=NumericalPlotter.GRID_COLOR)
        ax.legend(loc='best', facecolor=NumericalPlotter.BG_COLOR, 
                 edgecolor=NumericalPlotter.GRID_COLOR, labelcolor=NumericalPlotter.AXIS_COLOR)
        
        # Tick colors
        ax.tick_params(colors=NumericalPlotter.AXIS_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(NumericalPlotter.AXIS_COLOR)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_convergence_plot(
        convergence_history: List[float],
        title: str = "Convergence Analysis",
        y_label: str = "Error (log scale)"
    ) -> Figure:
        """
        Create convergence error plot (iterations vs error).
        
        Args:
            convergence_history: List of error values
            title: Plot title
            y_label: Y-axis label
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=NumericalPlotter.BG_COLOR)
        
        iterations = range(1, len(convergence_history) + 1)
        errors = convergence_history
        
        # Remove zero or negative values for log scale
        filtered_errors = [max(e, 1e-15) for e in errors]
        
        ax.semilogy(iterations, filtered_errors, 'b-o', linewidth=2.5, markersize=6, alpha=0.8)
        ax.grid(True, alpha=0.2, which='both', color=NumericalPlotter.GRID_COLOR)
        
        # Styling
        ax.set_facecolor(NumericalPlotter.BG_COLOR)
        ax.set_xlabel('Iteration', color=NumericalPlotter.AXIS_COLOR, fontsize=11)
        ax.set_ylabel(y_label, color=NumericalPlotter.AXIS_COLOR, fontsize=11)
        ax.set_title(title, color=NumericalPlotter.AXIS_COLOR, fontsize=13, fontweight='bold')
        ax.tick_params(colors=NumericalPlotter.AXIS_COLOR, labelsize=9)
        
        for spine in ax.spines.values():
            spine.set_color(NumericalPlotter.AXIS_COLOR)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_bisection_steps_plot(
        f: Callable[[float], float],
        steps_data: List[Tuple[float, float, float]],
        title: str = "Bisection Method - Step Visualization"
    ) -> Figure:
        """
        Create visualization of bisection method steps.
        
        Args:
            f: Function to plot
            steps_data: List of (a, b, c) tuples from each iteration
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        if not steps_data:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor=NumericalPlotter.BG_COLOR)
            ax.text(0.5, 0.5, "No step data available", ha='center', va='center',
                   transform=ax.transAxes, color='red')
            return fig
        
        # Extract bounds
        all_points = []
        for step in steps_data:
            all_points.extend(step)
        a_min, b_max = min(all_points), max(all_points)
        
        margin = (b_max - a_min) * 0.2
        a_plot = a_min - margin
        b_plot = b_max + margin
        
        fig, ax = plt.subplots(figsize=(12, 7), facecolor=NumericalPlotter.BG_COLOR)
        
        # Plot function
        x = np.linspace(a_plot, b_plot, 1000)
        try:
            y = np.array([f(xi) for xi in x])
            ax.plot(x, y, 'b-', linewidth=2.5, label='f(x)', alpha=0.9)
        except:
            pass
        
        ax.axhline(y=0, color=NumericalPlotter.GRID_COLOR, linestyle='--', alpha=0.5)
        ax.axvline(x=0, color=NumericalPlotter.GRID_COLOR, linestyle='--', alpha=0.5)
        
        # Color gradient for iterations
        colors = plt.cm.Greens(np.linspace(0.4, 1.0, len(steps_data)))
        
        for idx, (a, b, c) in enumerate(steps_data[-5:]):  # Show last 5 steps
            y_range = 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            offset = -idx * 0.15
            
            # Draw interval
            ax.plot([a, b], [offset, offset], 'o-', color=colors[idx], linewidth=2, markersize=8)
            ax.plot(c, offset, '*', color=colors[idx], markersize=20)
            ax.text(c, offset + 0.05, f'c{idx+1}', ha='center', color=colors[idx], fontsize=9)
        
        # Styling
        ax.set_facecolor(NumericalPlotter.BG_COLOR)
        ax.set_xlabel('x', color=NumericalPlotter.AXIS_COLOR, fontsize=11)
        ax.set_ylabel('f(x) / Step Offset', color=NumericalPlotter.AXIS_COLOR, fontsize=11)
        ax.set_title(title, color=NumericalPlotter.AXIS_COLOR, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2, color=NumericalPlotter.GRID_COLOR)
        ax.tick_params(colors=NumericalPlotter.AXIS_COLOR, labelsize=9)
        
        for spine in ax.spines.values():
            spine.set_color(NumericalPlotter.AXIS_COLOR)
        
        plt.tight_layout()
        return fig
