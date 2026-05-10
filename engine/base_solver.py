"""
Base Solver Class

Abstract base class for all numerical methods providing common
interface and step tracking functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any


@dataclass
class SolverStep:
    """Represents a single step in the numerical method."""
    
    step_number: int
    description: str
    values: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Return human-readable step description."""
        parts = [f"Step {self.step_number}: {self.description}"]
        for key, value in self.values.items():
            if isinstance(value, float):
                parts.append(f"  {key} = {value:.8f}")
            else:
                parts.append(f"  {key} = {value}")
        return "\n".join(parts)


@dataclass
class SolverResult:
    """Container for solver results and execution history."""
    
    root: float
    iterations: int
    tolerance_reached: bool
    error_message: str = ""
    steps: List[SolverStep] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if solver converged successfully."""
        return self.tolerance_reached and self.error_message == ""
    
    def __str__(self) -> str:
        """Return detailed result summary."""
        status = "✓ CONVERGED" if self.success else "✗ FAILED"
        summary = [
            f"\n{status}",
            f"Root: {self.root:.10f}",
            f"Iterations: {self.iterations}",
            f"Final Error: {self.convergence_history[-1]:.2e}" if self.convergence_history else "",
        ]
        if self.error_message:
            summary.append(f"Error: {self.error_message}")
        return "\n".join(filter(None, summary))


class BaseSolver(ABC):
    """
    Abstract base class for numerical solvers.
    
    All solver implementations must inherit from this class and implement
    the solve() method. Provides common utilities for tracking convergence
    and managing solver state.
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):
        """
        Initialize base solver.
        
        Args:
            tolerance: Convergence tolerance (default: 1e-6)
            max_iterations: Maximum iterations allowed (default: 100)
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self._steps: List[SolverStep] = []
        self._convergence_history: List[float] = []
    
    @abstractmethod
    def solve(self, *args, **kwargs) -> SolverResult:
        """
        Solve the numerical problem.
        
        Must be implemented by subclasses.
        """
        pass
    
    def _add_step(self, description: str, values: Dict[str, Any]) -> None:
        """
        Record a solver step for visualization.
        
        Args:
            description: Human-readable step description
            values: Dictionary of values at this step
        """
        step = SolverStep(
            step_number=len(self._steps) + 1,
            description=description,
            values=values
        )
        self._steps.append(step)
    
    def _add_convergence_value(self, error: float) -> None:
        """
        Track convergence metric over iterations.
        
        Args:
            error: Current error/tolerance measure
        """
        self._convergence_history.append(error)
    
    def _validate_interval(self, a: float, b: float, f: Callable) -> bool:
        """
        Validate that interval [a,b] contains a root.
        
        For methods requiring sign change, checks f(a)*f(b) < 0.
        
        Args:
            a: Left interval boundary
            b: Right interval boundary
            f: Function to evaluate
            
        Returns:
            True if interval is valid, False otherwise
        """
        fa = f(a)
        fb = f(b)
        return fa * fb < 0
    
    def _reset_history(self) -> None:
        """Clear step and convergence history for fresh solve."""
        self._steps = []
        self._convergence_history = []
