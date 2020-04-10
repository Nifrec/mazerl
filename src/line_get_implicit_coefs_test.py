"""
File to test the Line class of model.py

Author: Lulof Pirée
"""
import unittest
from record_types import Line
from distances import __get_line_implicit_coefs as get_line_implicit_coefs

class get_line_implicit_coefs_test(unittest.TestCase):

    def test_line_implicit_coefs_1(self):
        """
        Base case for the equation-coefficients computation.
        """
        line = Line(0, 0, 1, 1)
        # Expected values
        ex_a = 1
        ex_b = -1
        ex_c = 0
        a, b, c = get_line_implicit_coefs(line)
        assert(ex_a == a), f"a: expected: {ex_a}, got: {a}"
        assert(ex_b == b)
        assert(ex_c == c)

    def test_line_implicit_coefs_2(self):
        """
        Base case for the equation-coefficients computation.
        """
        line = Line(0, 4, 2, 0)
        # Expected values
        ex_a = -4 #x multiplier
        ex_b = -2 #y multiplier
        ex_c = 8
        a, b, c = get_line_implicit_coefs(line)
        assert(ex_a == a), f"a: expected: {ex_a}, got: {a}"
        assert(ex_b == b), f"b: expected: {ex_b}, got: {b}"
        assert(ex_c == c)

if (__name__ == "__main__"):
    unittest.main()