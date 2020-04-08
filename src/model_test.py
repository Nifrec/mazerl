"""
File to test the Model class of model.py

Author: Lulof Pirée
"""
import unittest
from model import Model, Size, Ball, MazeLayout, Line
import numpy as np

class ModelMovementTestCase(unittest.TestCase):
    """
    Basic movement without walls.
    """

    def setUp(self):
        self.size = Size(10000, 10000)
        self.ball_rad = 1
        self.start = np.array([50, 50])
        self.layout = MazeLayout(set([]), self.start, 
                np.array([99, 99]), self.size)
        self.model = Model(self.size, self.ball_rad)
        self.model.reset(self.layout)

    def compare_pos(self, expected, result):
        assert np.allclose(expected, result), \
            f"expected:{expected}, result:{result}"

    def test_movement_1(self):
        """
        Base case: no walls, only x-dir vel.
        """
        self.model.set_acceleration(1, 0)
        self.model.make_timestep()
        # Velocity should now be [1, 0]
        self.model.set_acceleration(0, 0)
        self.compare_pos(self.start + np.array([1, 0]),
                self.model.get_ball_position())
        self.model.make_timestep()
        # Velocity should remain be [1, 0]
        self.compare_pos(self.start + 2*np.array([1, 0]),
                self.model.get_ball_position())

    def test_movement_2(self):
        """
        Base case: no walls, only y-dir vel.
        """
        self.model.set_acceleration(0, 2)
        self.model.make_timestep()
        # Velocity should now be [0, 2]
        
        self.compare_pos(
                self.start + np.array([0, 2]),
                self.model.get_ball_position()
                )

        self.model.set_acceleration(0, 0)
        self.model.make_timestep()
        # Velocity should remain be [1, 0]
        self.compare_pos(
                self.start + 2*np.array([0, 2]),
                self.model.get_ball_position()
                )

    def test_movement_3(self):
        """
        Base case: no walls, negative acc.
        """
        self.model.set_acceleration(1, 1)
        self.model.make_timestep()
        # Velocity should now be [1, 1]
        self.compare_pos(
                self.start + np.array([1, 1]),
                self.model.get_ball_position()
                )

        self.model.set_acceleration(-2, -2)
        self.model.make_timestep()
        # Velocity should be [-1, -1]
        self.compare_pos(
                self.start,
                self.model.get_ball_position()
                )
        self.model.set_acceleration(0, 0)
        self.model.make_timestep()
        self.compare_pos(
                self.start + np.array([-1, -1]),
                self.model.get_ball_position()
                )

    def test_movement_4(self):
        """
        Base case: no walls, constant acc.
        """
        self.model.set_acceleration(1, 1)
        self.model.make_timestep()
        # Velocity should now be [1, 1]
        self.compare_pos(
                self.start + np.array([1, 1]),
                self.model.get_ball_position()
                )

        self.model.make_timestep()
        # Velocity should be [2, 2]
        self.compare_pos(
                self.start + np.array([3, 3]),
                self.model.get_ball_position()
                )
        # Velocity should be [3, 3]
        self.model.make_timestep()
        self.compare_pos(
                self.start + np.array([6, 6]),
                self.model.get_ball_position()
                )

class ModelBoundaryCollisionTestCase(unittest.TestCase):
    """
    Movement with boundary collisions.
    """

    def setUp(self):
        self.size = Size(100, 100)
        self.ball_rad = 1
        self.model = Model(self.size, self.ball_rad)

    def test_collision_boundary_1(self):
        """
        Sanity check: No wall hit.
        """
        layout = MazeLayout(set([]), np.array([2, 2]),
                np.array([99, 99]), self.size)
        self.model.reset(layout)

        assert (self.model.does_ball_hit_wall() == False)

    def test_collision_boundary_2(self):
        """
        Base case: left boundary.
        """
        layout = MazeLayout(set([]), np.array([2, 2]),
                np.array([99, 99]), self.size)
        self.model.reset(layout)
        self.model.set_acceleration(-1, 0)
        # New pos ball becomes (1, 2), and with rad=1 it will touch the border.
        self.model.make_timestep()

        assert (self.model.does_ball_hit_wall() == True)

    def test_collision_boundary_3(self):
        """
        Base case: top boundary.
        """
        layout = MazeLayout(set([]), np.array([2, 2]),
                np.array([99, 99]), self.size)
        self.model.reset(layout)
        self.model.set_acceleration(0, -1)
        # New pos ball becomes (2, 1), and with rad=1 it will touch the border.
        self.model.make_timestep()

        assert (self.model.does_ball_hit_wall() == True)

    def test_collision_boundary_4(self):
        """
        Base case: right boundary.
        """
        layout = MazeLayout(set([]), np.array([8, 8]),
                np.array([99, 99]), np.array([10, 10]))
        self.model.reset(layout)
        self.model.set_acceleration(1, 0)
        # New pos ball becomes (9, 8), and with rad=1 it will touch the border.
        self.model.make_timestep()

        assert (self.model.does_ball_hit_wall() == True)

    def test_collision_boundary_5(self):
        """
        Base case: bottom boundary.
        """
        layout = MazeLayout(set([]), np.array([8, 8]),
                np.array([99, 99]), np.array([10, 10]))
        self.model.reset(layout)
        self.model.set_acceleration(0, 1)
        # New pos ball becomes (8, 9), and with rad=1 it will touch the border.
        self.model.make_timestep()

        assert (self.model.does_ball_hit_wall() == True)

    def test_collision_boundary_6(self):
        """
        Corner case: completely off room boundary.
        """
        layout = MazeLayout(set([]), np.array([8, 8]),
                np.array([99, 99]), np.array([10, 10]))
        self.model.reset(layout)
        self.model.set_acceleration(100, 100) 
        # Will instantaneously fly from screen.
        self.model.make_timestep()

        assert (self.model.does_ball_hit_wall() == True)

    def test_collision_boundary_7(self):
        """
        Corner case (literally): diagonal movement (hit two at same time).
        """
        layout = MazeLayout(set([]), np.array([2, 2]),
                np.array([99, 99]), self.size)
        self.model.reset(layout)
        self.model.set_acceleration(-1, -1)
        # New pos ball becomes (1, 1), and with rad=1 it will touch the border.
        self.model.make_timestep()

        assert (self.model.does_ball_hit_wall() == True)

class ModelWallCollisionTestCase(unittest.TestCase):
    """
    Movement with wall collisions.
    """

    def setUp(self):
        self.size = Size(100, 100)
        self.ball_rad = 1
        self.model = Model(self.size, self.ball_rad)

    def test_collision_wall_1(self):
        """
        Sanity check: No wall hit, minimal distance.
        """
        wall1 = Line(4, 0, 4, 99)
        wall2 = Line(0, 4, 99, 4)
        layout = MazeLayout(set([wall1, wall2]), np.array([2, 2]),
                np.array([99, 99]), self.size)
        self.model.reset(layout)

        assert (self.model.does_ball_hit_wall() == False)

    def test_collision_wall_2(self):
        """
        Base case: hit right.
        """
        wall = Line(4, 0, 4, 99)
        layout = MazeLayout(set([wall]), np.array([2, 2]),
                np.array([99, 99]), self.size)
        self.model.reset(layout)
        self.model.set_acceleration(1, 0)
        # New pos ball becomes (3, 2), and with rad=1 it will touch the wall.
        self.model.make_timestep()

        assert (self.model.does_ball_hit_wall() == True)

    def test_collision_wall_3(self):
        """
        Base case: hit below.
        """
        wall2 = Line(0, 4, 99, 4)
        layout = MazeLayout(set([]), np.array([2, 2]),
                np.array([99, 99]), self.size)
        self.model.reset(layout)
        self.model.set_acceleration(0, 1)
        # New pos ball becomes (2, 3), and with rad=1 it will touch the wall.
        self.model.make_timestep()

        assert (self.model.does_ball_hit_wall() == True)

    def test_collision_wall_4(self):
        """
        Corner case: hit two walls at once.
        """
        wall1 = Line(4, 0, 4, 99)
        wall2 = Line(0, 4, 99, 4)
        layout = MazeLayout(set([wall1, wall2]), np.array([2, 2]),
                np.array([99, 99]), self.size)
        self.model.reset(layout)
        self.model.set_acceleration(1, 1)
        # New pos ball becomes (3, 3), and with rad=1 it will touch the walls.
        self.model.make_timestep()


        assert (self.model.does_ball_hit_wall() == True)

class ModelFinishCollisionTestCase(unittest.TestCase):
    """
    Checks if Model detects the endpoint correctly.
    """

    def setUp(self):
        self.size = Size(100, 100)
        self.ball_rad = 1
        self.model = Model(self.size, self.ball_rad)

    def test_collision_end_1(self):
        """
        Sanity check: not at end
        """
        layout = MazeLayout(set([]), np.array([2, 2]),
                np.array([99, 99]), self.size)
        self.model.reset(layout)
        self.model.set_acceleration(1, 1)
        # New pos ball becomes (3, 3), still far from (99, 99).
        self.model.make_timestep()

        assert (self.model.is_at_finish() == False)

    def test_collision_end_2(self):
        """
        Base case: ball reaches end
        """
        layout = MazeLayout(set([]), np.array([2, 2]),
                np.array([7, 2]), self.size)
        self.model.reset(layout)
        self.model.set_acceleration(2, 0)
        # New pos ball becomes (4, 2), not hit end yet.
        self.model.make_timestep()
        assert (self.model.is_at_finish() == False)
        self.model.set_acceleration(0, 0)
        # New pos ball becomes (6, 2), with rad=1 does hit it.
        self.model.make_timestep()
        assert (self.model.is_at_finish() == True)

if (__name__ == "__main__"):
    unittest.main()
