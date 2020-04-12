"""
File to test the classes in maze_generator.py

Author: Lulof Pirée
"""

# Library imports
import unittest
import numpy as np
from typing import Set
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

    def check_sets_of_lines_equal(self, set_0: Set[Line], set_1: Set[Line]):
        """
        Given two sets of Lines, checks for every line if there is a similar
        line in the other set.
        """
        
        while(set_0):
            line_0 = set_0.pop()
            has_counterpart = False
            for line_1 in set_1:
                if np.allclose(line_0.p0, line_1.p0) \
                        and np.allclose(line_0.p1, line_1.p1):
                    has_counterpart = True
                    set_1.remove(line_1)
                    break
            self.assertTrue(has_counterpart)
                    

    def test_set_up_1(self):
        """
        Base case: up-down path without a corner.
        """
        result = self.block.set_up(Direction.UP, Direction.DOWN)
        expected = set([self.left_wall, self.right_wall])
        print("MazeBlock: set_up 1:")
        print(f"Expected:{expected}, result: {result}")
        self.check_sets_of_lines_equal(expected, result)

    def test_set_up_2(self):
        """
        Base case: right-left path without a corner.
        """
        result = self.block.set_up(Direction.RIGHT, Direction.LEFT)
        expected = set([self.top_wall, self.bottom_wall])
        print("MazeBlock: set_up 2:")
        print(f"Expected:{expected}, result: {result}")
        self.check_sets_of_lines_equal(expected, result)

    def test_set_up_3(self):
        """
        Base case: left-to-bottom corner path.
        """
        result = self.block.set_up(Direction.LEFT, Direction.DOWN)
        expected = set([self.top_wall, self.right_wall])
        print("MazeBlock: set_up 3:")
        print(f"Expected:{expected}, result: {result}")
        self.check_sets_of_lines_equal(expected, result)

    def test_set_up_4(self):
        """
        Base case: right-to-top corner path.
        """
        result = self.block.set_up(Direction.RIGHT, Direction.UP)
        expected = set([self.left_wall, self.bottom_wall])
        print("MazeBlock: set_up 4:")
        print(f"Expected:{expected}, result: {result}")
        self.check_sets_of_lines_equal(expected, result)
        

if (__name__ == "__main__"):
    unittest.main()