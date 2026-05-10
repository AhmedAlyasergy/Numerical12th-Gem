"""
Unit Tests for Numerical Methods Workbench

Tests core functionality of all numerical solvers.
"""

import unittest
import numpy as np
from engine import BisectionMethod, SecantMethod, GaussSeidelMethod


class TestBisectionMethod(unittest.TestCase):
    """Test cases for Bisection Method."""
    
    def setUp(self):
        self.bisection = BisectionMethod(tolerance=1e-8, max_iterations=100)
    
    def test_simple_root(self):
        """Test finding root of x^2 - 4 in [1, 3]."""
        f = lambda x: x**2 - 4
        result = self.bisection.solve(f, 1, 3)
        
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.root, 2.0, places=5)
    
    def test_cubic_equation(self):
        """Test finding root of x^3 - 2 in [0, 2]."""
        f = lambda x: x**3 - 2
        result = self.bisection.solve(f, 0, 2)
        
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.root, 1.26, places=2)
    
    def test_no_sign_change(self):
        """Test that method rejects intervals without sign change."""
        f = lambda x: x**2 + 1  # Always positive
        result = self.bisection.solve(f, -5, 5)
        
        self.assertFalse(result.success)
        self.assertIn("Invalid interval", result.error_message)
    
    def test_convergence_tracking(self):
        """Test that convergence history is recorded."""
        f = lambda x: x**2 - 4
        result = self.bisection.solve(f, 1, 3)
        
        self.assertTrue(len(result.convergence_history) > 0)
        # Each iteration should reduce error
        for i in range(1, len(result.convergence_history)):
            self.assertLessEqual(
                result.convergence_history[i],
                result.convergence_history[i-1]
            )
    
    def test_step_recording(self):
        """Test that steps are recorded."""
        f = lambda x: x**2 - 4
        result = self.bisection.solve(f, 1, 3, tolerance=1e-4)
        
        self.assertTrue(len(result.steps) > 0)


class TestSecantMethod(unittest.TestCase):
    """Test cases for Secant Method."""
    
    def setUp(self):
        self.secant = SecantMethod(tolerance=1e-8, max_iterations=100)
    
    def test_simple_root(self):
        """Test finding root of x^3 - 2 with x0=0, x1=2."""
        f = lambda x: x**3 - 2
        result = self.secant.solve(f, 0, 2)
        
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.root, 1.26, places=2)
    
    def test_quadratic(self):
        """Test finding root of x^2 - 5."""
        f = lambda x: x**2 - 5
        result = self.secant.solve(f, 1, 3)
        
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.root, 2.236, places=2)
    
    def test_convergence_faster_than_bisection(self):
        """Test that secant converges in fewer or equal iterations."""
        f = lambda x: x**3 - 2
        
        bisection = BisectionMethod(tolerance=1e-8)
        secant = SecantMethod(tolerance=1e-8)
        
        bisect_result = bisection.solve(f, 0, 2)
        secant_result = secant.solve(f, 0, 2)
        
        # Secant should converge in fewer iterations
        self.assertLessEqual(secant_result.iterations, bisect_result.iterations)


class TestGaussSeidelMethod(unittest.TestCase):
    """Test cases for Gauss-Seidel Method."""
    
    def setUp(self):
        self.gauss_seidel = GaussSeidelMethod(tolerance=1e-8, max_iterations=100)
    
    def test_diagonally_dominant_system(self):
        """Test 3x3 diagonally dominant system."""
        A = np.array([
            [4, -1, 0],
            [-1, 4, -1],
            [0, -1, 3]
        ], dtype=float)
        b = np.array([15, 10, 10], dtype=float)
        
        result = self.gauss_seidel.solve(A, b)
        
        self.assertTrue(result.success)
        # Verify solution
        solution = np.array([4.005, 2.997, 4.001])
        residual = np.linalg.norm(A @ result.convergence_history[-1] - b)
        self.assertLess(residual, 1e-4)
    
    def test_2x2_system(self):
        """Test simple 2x2 system."""
        A = np.array([
            [2, 1],
            [1, 3]
        ], dtype=float)
        b = np.array([3, 4], dtype=float)
        
        result = self.gauss_seidel.solve(A, b)
        
        self.assertTrue(result.success)
    
    def test_singular_matrix_detection(self):
        """Test that singular matrices are detected."""
        A = np.array([
            [1, 2],
            [2, 4]  # Singular (row 2 = 2 * row 1)
        ], dtype=float)
        b = np.array([3, 6], dtype=float)
        
        result = self.gauss_seidel.solve(A, b)
        
        self.assertFalse(result.success)
        self.assertIn("singular", result.error_message.lower())
    
    def test_non_square_matrix(self):
        """Test that non-square matrices are rejected."""
        A = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ], dtype=float)
        b = np.array([3, 4], dtype=float)
        
        result = self.gauss_seidel.solve(A, b)
        
        self.assertFalse(result.success)


class TestIntegration(unittest.TestCase):
    """Integration tests across all methods."""
    
    def test_all_methods_converge_to_same_root(self):
        """Test that different methods find the same root."""
        f = lambda x: x**3 - x - 1
        
        bisection = BisectionMethod()
        secant = SecantMethod()
        
        bisect_result = bisection.solve(f, 1, 2)
        secant_result = secant.solve(f, 1, 2)
        
        self.assertAlmostEqual(bisect_result.root, secant_result.root, places=5)
    
    def test_tighter_tolerance_more_iterations(self):
        """Test that tighter tolerance requires more iterations."""
        f = lambda x: x**2 - 4
        
        loose = BisectionMethod(tolerance=1e-4)
        tight = BisectionMethod(tolerance=1e-10)
        
        loose_result = loose.solve(f, 1, 3)
        tight_result = tight.solve(f, 1, 3)
        
        self.assertLessEqual(loose_result.iterations, tight_result.iterations)


if __name__ == "__main__":
    unittest.main()
