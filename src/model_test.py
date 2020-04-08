"""
File to test the Model class of model.py

Author: Lulof Pirée
"""
import unittest
from model import Model, Size, Ball, MazeLayout
import numpy as np

class ModelMovementTestCase(unittest.TestCase):

    def setUp(self):
        self.size = Size(10000, 10000)
        self.ball_rad = 1
        self.start = np.array([50, 50])
        self.layout = MazeLayout(set([]), self.start, 
                np.array([1000, 1000]), self.size)
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

if (__name__ == "__main__"):
    unittest.main()
