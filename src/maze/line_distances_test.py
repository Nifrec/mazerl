"""
File to test several auxiliary functions involved with computing
distances and intersections of lines.

Author: Lulof Pirée
"""
import unittest
import numpy as np
import math

from record_types import Line
from distances import Orientation
from distances import do_lines_intersect
from distances import __compute_orientation_points as compute_orientation_points
from distances import dist_line_to_line

class DoLinesIntersectTestCase(unittest.TestCase):
    """
    Tests the check-intersection-of-two-line-segments-function.
    """

    def test_intersect_1(self):
        """
        Base case: no intersection.
        """
        line_0 = Line(0, 0, 10, 0)
        line_1 = Line(1, 1, 100, 100)

        self.assertFalse(do_lines_intersect(line_0, line_1))

    def test_intersect_2(self):
        """
        Corner case: just intersect at endpoint.
        """
        line_0 = Line(0, 0, 10, 0)
        line_1 = Line(0, 0, 100, 100)

        self.assertTrue(do_lines_intersect(line_0, line_1))

    def test_intersect_3(self):
        """
        Base case: intersect halfaway.
        """
        line_0 = Line(5, 5, 15, 15)
        line_1 = Line(5, 15, 15, 5)

        self.assertTrue(do_lines_intersect(line_0, line_1))

    def test_intersect_4(self):
        """
        Base case: collinear, do intersect.
        """
        line_0 = Line(5, 5, 15, 15)
        line_1 = Line(3, 3, 12, 12)

        self.assertTrue(do_lines_intersect(line_0, line_1))

    def test_intersect_5(self):
        """
        Base case: part same inf line, but no intersection.
        """
        line_0 = Line(0, 0, 3, 3)
        line_1 = Line(4.3, 4.3, 12, 12)

        self.assertFalse(do_lines_intersect(line_0, line_1))

    def test_intersect_6(self):
        """
        Base case: collinear, do not intersect.
        """
        line_0 = Line(5, 5, 15, 15)
        line_1 = Line(5, 4, 15, 14)

        self.assertFalse(do_lines_intersect(line_0, line_1))


class OrientationLinesTestCase(unittest.TestCase):
    """
    Tests __compute_orientation_points function.
    NOTE: when using pygame, the y-axis is mirrored (y=0 is at top of screen).
    This does not influence whether lines are crossing or not.
    """

    def test_orientation_1(self):
        """
        Base case: simple clockwise triangle.
        """
        line = Line(0, 0, 1, 1)
        p = np.array([2, 0])
        output = compute_orientation_points(line, p)

        self.assertEqual(Orientation.CLOCKWISE, output)

    def test_orientation_2(self):
        """
        Base case: simple counterclockwise triangle.
        """
        line = Line(0, 0, 1, 0)
        p = np.array([3, 2])
        output = compute_orientation_points(line, p)

        self.assertEqual(Orientation.COUNTERCLOCKWISE, output)

    def test_orientation_3(self):
        """
        Base case: simple collinear line.
        """
        line = Line(0, 0, 5, 5)
        p = np.array([2, 2])
        output = compute_orientation_points(line, p)

        self.assertEqual(Orientation.COLLINEAR, output)

    def test_orientation_4(self):
        """
        Base case: right-angle clockwise triangle.
        """
        line = Line(0, 0, 1, 1)
        p = np.array([2, 1])
        output = compute_orientation_points(line, p)

        self.assertEqual(Orientation.CLOCKWISE, output)

    def test_orientation_5(self):
        """
        Base case: wide-angle clockwise triangle.
        """
        line = Line(1, 1, 2, 1)
        p = np.array([3, 0])
        output = compute_orientation_points(line, p)

        self.assertEqual(Orientation.CLOCKWISE, output)

    def test_orientation_6(self):
        """
        Base case: right-angle counterclockwise triangle.
        """
        line = Line(1, 1, 2, 1)
        p = np.array([2, 2])
        output = compute_orientation_points(line, p)

        self.assertEqual(Orientation.COUNTERCLOCKWISE, output)

    def test_orientation_7(self):
        """
        Corner case: clockwise, 2 points same x-val.
        """
        line = Line(1, 1, 1, 2)
        p = np.array([2, 2])
        output = compute_orientation_points(line, p)

        self.assertEqual(Orientation.CLOCKWISE, output)

    def test_orientation_8(self):
        """
        Corner case: counterclockwise, 2 points same x-val.
        """
        line = Line(1, 1, 1, 2)
        p = np.array([0, 2])
        output = compute_orientation_points(line, p)

        self.assertEqual(Orientation.COUNTERCLOCKWISE, output)

class DistLineToLineTestCase(unittest.TestCase):
    """
    Tests dist_line_to_line().
    """
    def test_dist_lines_1(self):
        """
        Base case: intersecting lines.
        """
        line_0 = Line(1, 1, 2, 2)
        line_1 = Line(1, 2, 2, 1)
        output = dist_line_to_line(line_0, line_1)
        expected = 0.0

        self.assertAlmostEqual(expected, output)

    def test_dist_lines_2(self):
        """
        Base case: colinear lines.
        """
        line_0 = Line(0, 1, 1, 2)
        line_1 = Line(0, 0, 1, 1)
        output = dist_line_to_line(line_0, line_1)
        expected = 0.5*math.sqrt(1**2 + 1**2)

        self.assertAlmostEqual(expected, output)

    def test_dist_lines_3(self):
        """
        Base case: colinear lines on same infinite line.
        """
        line_0 = Line(0, 0, 2, 2)
        line_1 = Line(6, 6, 5, 5)
        output = dist_line_to_line(line_0, line_1)
        expected = math.sqrt(3**2 + 3**2)

        self.assertAlmostEqual(expected, output)

    def test_dist_lines_4(self):
        """
        Base case: shortest dist is to endpoint of one of the two.
        """
        # Note: line_0 not noted left-to-right. It should handle this.
        line_0 = Line(5, 4, 2, 1) 
        line_1 = Line(1, 6, 10, 6)
        output = dist_line_to_line(line_0, line_1)
        expected = 2

        self.assertAlmostEqual(expected, output)

if (__name__ == "__main__"):
    unittest.main()