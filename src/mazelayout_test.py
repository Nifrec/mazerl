"""
File to test the MazeLayout class of model.py

Author: Lulof Pirée
"""
import unittest
from model import Ball, Line, MazeLayout

class MazeLayoutTestCase(unittest.TestCase):


    def test_get_ball_hits_wall_1(self):
        """
        Base case: normal ball hits normal wall.
        """
        lines = set([Line(0, 3, 3, 0)])
        ball = Ball(0, 0, 2.5)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == True)

    def test_get_ball_hits_wall_2(self):
        """
        Base case: normal ball does not hit a wall.
        """
        lines = set([Line(0, 3, 3, 0)])
        ball = Ball(0, 0, 1)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == False)

    def test_get_ball_hits_wall_3(self):
        """
        Corner case: normal ball just hits a wall.
        """
        lines = set([Line(3, 3, 3, 0)])
        ball = Ball(0, 1.5, 3)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == True)

    def test_get_ball_hits_wall_4(self):
        """
        Base case: 2 walls, no hit.
        """
        lines = set([Line(0, 3, 3, 0), Line(-4, -6, 3, 0)])
        ball = Ball(5, 6, 1)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == False)

    def test_get_ball_hits_wall_5(self):
        """
        Base case: 2 walls, one hits.
        """
        lines = set([Line(0, 3, 3, 0), Line(-4, -6, 3, 0)])
        ball = Ball(-4, -5, 2)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == True)

    def test_get_ball_hits_wall_6(self):
        """
        Corner case: endpoints of line are far away.
        """
        lines = set([Line(-5, -5, 5, 5)])
        ball = Ball(0, 0, 1)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == True)

    def test_get_ball_hits_wall_7(self):
        """
        Corner case: Does not hit but would do so if the line were longer.
        """
        lines = set([Line(-5, -5, -3, -3)])
        ball = Ball(0, 0, 1)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == False)

    def test_get_ball_hits_wall_8(self):
        """
        Corner case: Does not hit but would do so if the line were longer v2.
        """
        lines = set([Line(-5, 5, 0, 0)])
        ball = Ball(5, -5, 1)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == False)

    def test_get_ball_hits_wall_9(self):
        """
        Corner case: Does not hit but would do so if the line were longer v3.
        """
        lines = set([Line(1, 5, 1, 4)])
        ball = Ball(1, 0, 1)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == False)

    def test_get_ball_hits_wall_10(self):
        """
        Corner case: Does not hit but would do so if the line were longer v4.
        """
        lines = set([Line(1, 10, -5, 10)])
        ball = Ball(6, 10, 2)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == False)

    def test_get_ball_hits_wall_11(self):
        """
        Corner case: Hits, both points of line left of ball center.
        """
        lines = set([Line(0, 100, 0, -100)])
        ball = Ball(1, 0, 3)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == True)

    def test_get_ball_hits_wall_12(self):
        """
        Corner case: Hits, both points of line right of ball center.
        """
        lines = set([Line(0, 100, 0, -100)])
        ball = Ball(-1, 0, 3)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == True)

    def test_get_ball_hits_wall_13(self):
        """
        Corner case: coordinates line points all smaller than
        ball coordinate values (of respective dims).
        """
        lines = set([Line(0, -3, -3, 0)])
        ball = Ball(0, 0, 2.5)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == True)

    def test_get_ball_hits_wall_14(self):
        """
        Corner case: Does not hit but would do so if the line were longer v5:
        One point bigger y and one point smaller y than ball.
        """
        lines = set([Line(-100, -0.1, -5, 0.1)])
        ball = Ball(0, 0, 4)
        layout = MazeLayout(lines)

        assert (layout.get_ball_hits_wall(ball) == False)

if (__name__ == "__main__"):
    unittest.main()