"""
Gauss-Seidel Method Implementation

An iterative method for solving systems of linear equations,
using NumPy for efficient matrix operations.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
from .base_solver import BaseSolver, SolverResult


class GaussSeidelMethod(BaseSolver):
    """
    Gauss-Seidel Method Solver
    
    Solves systems of linear equations Ax = b iteratively using updated
    values immediately as they become available.
    
    Mathematical Basis:
    - Iterative refinement of solution vector
    - Generally converges faster than Jacobi method
    - Requires diagonal dominance for guaranteed convergence
    - Updates use most recent available values
    """
    
    def _check_diagonal_dominance(self, A: np.ndarray) -> bool:
        """
        Check if matrix has strict diagonal dominance.
        
        Diagonal dominance (sufficient but not necessary for convergence):
        |A[i,i]| > sum(|A[i,j]| for j != i)
        
        Args:
            A: Coefficient matrix
            
        Returns:
            True if matrix is strictly diagonally dominant
        """
        n = A.shape[0]
        for i in range(n):
            diag = abs(A[i, i])
            off_diag_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
            if diag <= off_diag_sum:
                return False
        return True
    
    def _check_convergence_condition(self, A: np.ndarray) -> Tuple[bool, str]:
        """
        Check necessary convergence conditions.
        
        Args:
            A: Coefficient matrix
            
        Returns:
            Tuple of (is_valid, warning_message)
        """
        # Check dimensions
        if A.shape[0] != A.shape[1]:
            return False, "Matrix must be square."
        
        # Check for singular matrix
        try:
            det = np.linalg.det(A)
            if abs(det) < 1e-15:
                return False, "Matrix is singular or nearly singular."
        except np.linalg.LinAlgError:
            return False, "Matrix is singular."
        
        # Check diagonal dominance (sufficient condition)
        if not self._check_diagonal_dominance(A):
            return True, ("Warning: Matrix is not strictly diagonally dominant. "
                         "Convergence is not guaranteed, but may still occur.")
        
        return True, ""
    
    def solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x0: np.ndarray = None,
        tolerance: float = None,
        max_iterations: int = None
    ) -> SolverResult:
        """
        Solve system Ax = b using Gauss-Seidel iteration.
        
        Args:
            A: Coefficient matrix (n x n)
            b: Right-hand side vector (n,)
            x0: Initial guess (uses zeros if None)
            tolerance: Convergence tolerance (uses self.tolerance if None)
            max_iterations: Max iterations (uses self.max_iterations if None)
            
        Returns:
            SolverResult containing solution vector, iterations, and steps
        """
        self._reset_history()
        
        # Validate and prepare inputs
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).flatten()
        
        n = A.shape[0]
        
        # Check convergence conditions
        is_valid, warning = self._check_convergence_condition(A)
        if not is_valid:
            return SolverResult(
                root=0.0,  # Placeholder
                iterations=0,
                tolerance_reached=False,
                error_message=warning,
                steps=self._steps,
                convergence_history=self._convergence_history
            )
        
        # Initialize solution vector
        x = x0 if x0 is not None else np.zeros(n)
        x = np.asarray(x, dtype=float).flatten()
        
        if x.shape[0] != n:
            return SolverResult(
                root=0.0,
                iterations=0,
                tolerance_reached=False,
                error_message=f"Initial guess dimension {x.shape[0]} does not match matrix dimension {n}.",
                steps=self._steps,
                convergence_history=self._convergence_history
            )
        
        tol = tolerance or self.tolerance
        max_iter = max_iterations or self.max_iterations
        
        self._add_step(
            "Initialize Gauss-Seidel iteration",
            {
                "system_size": n,
                "initial_guess": str(x),
                "tolerance": tol,
                "max_iterations": max_iter,
                "diagonal_dominant": self._check_diagonal_dominance(A)
            }
        )
        
        for iteration in range(1, max_iter + 1):
            x_old = x.copy()
            
            # Gauss-Seidel update: use most recent values immediately
            for i in range(n):
                sum_val = b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x_old[i+1:])
                if abs(A[i, i]) < 1e-15:
                    return SolverResult(
                        root=0.0,
                        iterations=iteration,
                        tolerance_reached=False,
                        error_message=f"Diagonal element A[{i},{i}] is near zero.",
                        steps=self._steps,
                        convergence_history=self._convergence_history
                    )
                x[i] = sum_val / A[i, i]
            
            # Compute error (Euclidean norm of difference)
            error = np.linalg.norm(x - x_old)
            self._add_convergence_value(error)
            
            residual = np.linalg.norm(A @ x - b)
            
            self._add_step(
                f"Iteration {iteration}",
                {
                    "solution": str(np.round(x, 8)),
                    "error": error,
                    "residual": residual,
                    "max_change": np.max(np.abs(x - x_old))
                }
            )
            
            # Check convergence
            if error < tol:
                return SolverResult(
                    root=float(np.linalg.norm(x)),  # Use norm as scalar representation
                    iterations=iteration,
                    tolerance_reached=True,
                    steps=self._steps,
                    convergence_history=self._convergence_history
                )
        
        # Max iterations reached
        return SolverResult(
            root=float(np.linalg.norm(x)),
            iterations=max_iter,
            tolerance_reached=False,
            error_message=f"Maximum iterations ({max_iter}) reached without convergence.",
            steps=self._steps,
            convergence_history=self._convergence_history
        )
