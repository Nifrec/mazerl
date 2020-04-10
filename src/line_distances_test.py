"""
File to test several auxiliary functions involved with computing
distances and intersections of lines.

Author: Lulof Pirée
"""
import unittest
import numpy as np
from record_types import Line
from distances import Orientation, do_lines_intersect
from distances import __compute_orientation_points as compute_orientation_points

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
        Base case: collinear, no intersect.
        """
        line_0 = Line(5, 5, 15, 15)
        line_1 = Line(3, 3, 12, 12)

        self.assertFalse(do_lines_intersect(line_0, line_1))

    def test_intersect_5(self):
        """
        Base case: part same inf line, but no intersection.
        """
        line_0 = Line(0, 0, 3, 3)
        line_1 = Line(4.3, 4.3, 12, 12)

        self.assertFalse(do_lines_intersect(line_0, line_1))


class OrientationLinesTestCase(unittest.TestCase):
    """
    Tests __compute_orientation_points function.
    """

    def test_orientation_1(self):
        """
        Base case: simple clockwise triangle.
        """
        p0 = np.array([0, 0])
        p1 = np.array([1, 1])
        p2 = np.array([2, 0])
        output = compute_orientation_points(p0, p1, p2)

        self.assertEqual(Orientation.CLOCKWISE, output)

    def test_orientation_2(self):
        """
        Base case: simple counterclockwise triangle.
        """
        p0 = np.array([0, 0])
        p1 = np.array([-1, -1])
        p2 = np.array([2, 0])
        output = compute_orientation_points(p0, p1, p2)

        self.assertEqual(Orientation.COUNTERCLOCKWISE, output)

    def test_orientation_3(self):
        """
        Base case: simple collinear line.
        """
        p0 = np.array([0, 0])
        p1 = np.array([5, 5])
        p2 = np.array([2, 2])
        output = compute_orientation_points(p0, p1, p2)

        self.assertEqual(Orientation.COLLINEAR, output)

    def test_orientation_4(self):
        """
        Base case: right-angle clockwise triangle.
        """
        p0 = np.array([0, 0])
        p1 = np.array([1, 1])
        p2 = np.array([2, 1])
        output = compute_orientation_points(p0, p1, p2)

        self.assertEqual(Orientation.CLOCKWISE, output)

    def test_orientation_5(self):
        """
        Base case: wide-angle counterclockwise triangle.
        """
        p0 = np.array([1, 1])
        p1 = np.array([2, 1])
        p2 = np.array([3, 0])
        output = compute_orientation_points(p0, p1, p2)

        self.assertEqual(Orientation.COUNTERCLOCKWISE, output)

    def test_orientation_6(self):
        """
        Base case: right-angle counterclockwise triangle.
        """
        p0 = np.array([1, 1])
        p1 = np.array([2, 1])
        p2 = np.array([2, 0])
        output = compute_orientation_points(p0, p1, p2)

        self.assertEqual(Orientation.COUNTERCLOCKWISE, output)




if (__name__ == "__main__"):
    unittest.main()