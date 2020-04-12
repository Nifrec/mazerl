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
from maze_generator import Direction, MazeBlock

class MazeBlockTestCase(unittest.TestCase):

    def setUp(self):
        self.block = MazeBlock(10, 10, 15, 15)

    def test_left_wall(self):
        wall = self.block.generate_wall(Direction.LEFT)
        expected = Line(10, 10, 10, 25)
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

    def test_right_wall(self):
        wall = self.block.generate_wall(Direction.RIGHT)
        expected = Line(25, 10, 25, 25)
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

    def test_top_wall(self):
        wall = self.block.generate_wall(Direction.UP)
        expected = Line(10, 10, 25, 10)
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

    def test_bottom_wall(self):
        wall = self.block.generate_wall(Direction.DOWN)
        expected = Line(10, 25, 25, 25)
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

if (__name__ == "__main__"):
    unittest.main()