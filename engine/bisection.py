"""
Bisection Method Implementation

A robust root-finding algorithm that repeatedly bisects an interval
and selects the subinterval where the root must lie.
"""

from typing import Callable
from .base_solver import BaseSolver, SolverResult


class BisectionMethod(BaseSolver):
    """
    Bisection Method Solver
    
    Finds roots of continuous functions by repeatedly halving an interval
    where a root is guaranteed to exist (sign change).
    
    Mathematical Basis:
    - Requires: f(a) * f(b) < 0 (sign change)
    - Convergence: Linear, always converges
    - Each iteration reduces error by factor of 2
    """
    
    def solve(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        tolerance: float = None,
        max_iterations: int = None
    ) -> SolverResult:
        """
        Find root using bisection method.
        
        Args:
            f: Function to find root for
            a: Left interval boundary
            b: Right interval boundary
            tolerance: Convergence tolerance (uses self.tolerance if None)
            max_iterations: Max iterations (uses self.max_iterations if None)
            
        Returns:
            SolverResult containing root, iterations, and execution steps
            
        Raises:
            ValueError: If interval does not contain a root (f(a)*f(b) >= 0)
        """
        self._reset_history()
        tol = tolerance or self.tolerance
        max_iter = max_iterations or self.max_iterations
        
        # Validate interval
        if not self._validate_interval(a, b, f):
            return SolverResult(
                root=0.0,
                iterations=0,
                tolerance_reached=False,
                error_message=f"Invalid interval: f({a:.6f})*f({b:.6f}) >= 0. "
                              f"Function must have opposite signs at boundaries."
            )
        
        # Ensure a < b
        if a > b:
            a, b = b, a
        
        fa = f(a)
        fb = f(b)
        
        self._add_step(
            "Initial interval validation",
            {"a": a, "b": b, "f(a)": fa, "f(b)": fb}
        )
        
        for iteration in range(1, max_iter + 1):
            # Compute midpoint
            c = (a + b) / 2
            fc = f(c)
            error = abs(b - a) / 2
            
            self._add_convergence_value(error)
            self._add_step(
                f"Bisect interval",
                {
                    "a": a,
                    "b": b,
                    "c": c,
                    "f(c)": fc,
                    "error": error,
                    "interval_width": b - a
                }
            )
            
            # Check convergence
            if error < tol or fc == 0:
                return SolverResult(
                    root=c,
                    iterations=iteration,
                    tolerance_reached=True,
                    steps=self._steps,
                    convergence_history=self._convergence_history
                )
            
            # Update interval
            if fa * fc < 0:
                # Root in [a, c]
                b = c
                fb = fc
            else:
                # Root in [c, b]
                a = c
                fa = fc
        
        # Max iterations reached
        c = (a + b) / 2
        return SolverResult(
            root=c,
            iterations=max_iter,
            tolerance_reached=False,
            error_message=f"Maximum iterations ({max_iter}) reached without convergence.",
            steps=self._steps,
            convergence_history=self._convergence_history
        )
