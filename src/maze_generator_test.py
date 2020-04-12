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
        self.left_wall = Line(10, 10, 10, 25)
        self.right_wall = Line(25, 10, 25, 25)
        self.top_wall = Line(10, 10, 25, 10)
        self.bottom_wall =  Line(10, 25, 25, 25)

    def test_left_wall(self):
        wall = self.block.generate_wall(Direction.LEFT)
        expected = self.left_wall
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

    def test_right_wall(self):
        wall = self.block.generate_wall(Direction.RIGHT)
        expected = self.right_wall
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

    def test_top_wall(self):
        wall = self.block.generate_wall(Direction.UP)
        expected = self.top_wall
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

    def test_bottom_wall(self):
        wall = self.block.generate_wall(Direction.DOWN)
        expected = self.bottom_wall
        self.assertTrue(np.allclose(wall.p0, expected.p0))
        self.assertTrue(np.allclose(wall.p1, expected.p1))

    def test_walls_with_path(self):
        pass

if (__name__ == "__main__"):
    unittest.main()