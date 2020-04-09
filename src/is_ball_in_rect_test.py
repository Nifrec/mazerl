"""
File to test the auxiliary function model.is_ball_in_rect().

Author: Lulof Pirée
"""
from model import Ball, Size
from distances import is_ball_in_rect
import unittest

class BallInRectTestCase(unittest.TestCase):
    """
    Checks the auxiliary function model.is_ball_in_rect()
    """

    def test_is_ball_in_rect_1(self):
        """
        Base case: ball in rect.
        """
        ball = Ball(5, 6, 2)
        rect = Size(10, 10)

        assert (is_ball_in_rect(ball, rect) == True)

    def test_is_ball_in_rect_2(self):
        """
        Base case: ball hits top.
        """
        ball = Ball(5, 2, 2)
        rect = Size(10, 10)

        assert (is_ball_in_rect(ball, rect) == False)

    def test_is_ball_in_rect_3(self):
        """
        Base case: ball overlaps left.
        """
        ball = Ball(1, 5, 3)
        rect = Size(10, 10)

        assert (is_ball_in_rect(ball, rect) == False)

    def test_is_ball_in_rect_4(self):
        """
        Base case: ball hits right.
        """
        ball = Ball(1, 5, 9)
        rect = Size(10, 10)

        assert (is_ball_in_rect(ball, rect) == False)


if (__name__ == "__main__"):
    unittest.main()