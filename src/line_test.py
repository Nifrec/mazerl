"""
File to test the Line class of model.py

Author: Lulof Pirée
"""
import unittest
from model import Line

class LineTestCase(unittest.TestCase):

    def test_abc1(self):
        """
        Base case for the equation-coefficients computation.
        """
        line = Line(0, 0, 1, 1)
        # Expected values
        ex_a = 1
        ex_b = -1
        ex_c = 0

        assert(ex_a == line.a), f"a: expected: {ex_a}, got: {line.a}"
        assert(ex_b == line.b)
        assert(ex_c == line.c)

    def test_abc2(self):
        """
        Base case for the equation-coefficients computation.
        """
        line = Line(0, 4, 2, 0)
        # Expected values
        ex_a = -4 #x multiplier
        ex_b = -2 #y multiplier
        ex_c = 8

        assert(ex_a == line.a), f"a: expected: {ex_a}, got: {line.a}"
        assert(ex_b == line.b), f"b: expected: {ex_b}, got: {line.b}"
        assert(ex_c == line.c)

if (__name__ == "__main__"):
    unittest.main()