"""
Secant Method Implementation

A root-finding algorithm that uses secant lines instead of tangent lines,
eliminating the need for derivative computation.
"""

from typing import Callable
from .base_solver import BaseSolver, SolverResult


class SecantMethod(BaseSolver):
    """
    Secant Method Solver
    
    Finds roots using two initial points to form secant lines that
    approximate the tangent. Does not require derivative computation.
    
    Mathematical Basis:
    - Uses finite difference to approximate derivative
    - Convergence: Super-linear (~1.618 order)
    - More efficient than bisection but less robust
    """
    
    def solve(
        self,
        f: Callable[[float], float],
        x0: float,
        x1: float,
        tolerance: float = None,
        max_iterations: int = None
    ) -> SolverResult:
        """
        Find root using secant method.
        
        Args:
            f: Function to find root for
            x0: First initial point
            x1: Second initial point
            tolerance: Convergence tolerance (uses self.tolerance if None)
            max_iterations: Max iterations (uses self.max_iterations if None)
            
        Returns:
            SolverResult containing root, iterations, and execution steps
        """
        self._reset_history()
        tol = tolerance or self.tolerance
        max_iter = max_iterations or self.max_iterations
        
        f_x0 = f(x0)
        f_x1 = f(x1)
        
        self._add_step(
            "Initialize with two points",
            {
                "x0": x0,
                "x1": x1,
                "f(x0)": f_x0,
                "f(x1)": f_x1,
                "difference": abs(x1 - x0)
            }
        )
        
        for iteration in range(1, max_iter + 1):
            # Check for zero denominator (parallel to x-axis)
            denominator = f_x1 - f_x0
            if abs(denominator) < 1e-15:
                return SolverResult(
                    root=x1,
                    iterations=iteration,
                    tolerance_reached=False,
                    error_message="Secant line nearly parallel to x-axis. "
                                  "Function values too close or derivative near zero.",
                    steps=self._steps,
                    convergence_history=self._convergence_history
                )
            
            # Compute next point
            x2 = x1 - f_x1 * (x1 - x0) / denominator
            f_x2 = f(x2)
            error = abs(x2 - x1)
            
            self._add_convergence_value(error)
            self._add_step(
                "Compute secant intersection",
                {
                    "x0": x0,
                    "x1": x1,
                    "x2": x2,
                    "f(x2)": f_x2,
                    "error": error,
                    "slope": denominator / (x1 - x0) if x1 != x0 else 0
                }
            )
            
            # Check convergence
            if error < tol or f_x2 == 0:
                return SolverResult(
                    root=x2,
                    iterations=iteration,
                    tolerance_reached=True,
                    steps=self._steps,
                    convergence_history=self._convergence_history
                )
            
            # Update points
            x0, f_x0 = x1, f_x1
            x1, f_x1 = x2, f_x2
        
        # Max iterations reached
        return SolverResult(
            root=x1,
            iterations=max_iter,
            tolerance_reached=False,
            error_message=f"Maximum iterations ({max_iter}) reached without convergence.",
            steps=self._steps,
            convergence_history=self._convergence_history
        )
