"""
File to test the classes in maze_generator.py

Author: Lulof Pirée
"""

# Library imports
import unittest
import numpy as np
# Local imports
from model import MazeLayout
from record_types import Line, Size
from maze_generator import MazeBlock

class MazeBlockTestCase(unittest.TestCase):

    def setUp(self):
        self.block = MazeBlock(10, 10, 15, 15)

    def test_left_wall(self):
        wall = self.block.generate_wall_left()
        expected = Line(10, 10, 10, 25)
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

    def test_right_wall(self):
        wall = self.block.generate_wall_right()
        expected = Line(25, 10, 25, 25)
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

    def test_top_wall(self):
        wall = self.block.generate_wall_top()
        expected = Line(10, 10, 25, 10)
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

    def test_top_bottom(self):
        wall = self.block.generate_wall_bottom()
        expected = Line(10, 25, 25, 25)
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

if (__name__ == "__main__"):
    unittest.main()