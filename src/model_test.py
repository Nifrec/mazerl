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
        self.ball_size = 1
        self.start = np.array([50, 50])
        self.layout = MazeLayout(set([]), self.start, 
                np.array([1000, 1000]), self.size)
        self.model = Model(self.size, self.ball_size)
        self.model.reset(self.layout)

    def test_movement_1(self):
        """
        Base case: no walls, only x-dir vel.
        """
        self.model.set_acceleration(1, 0)
        self.model.make_timestep()
        # Velocity should now be [1, 0]
        self.model.set_acceleration(0, 0)
        assert np.allclose(self.model.get_ball_position,
                self.start + np.array([1, 0]))
        self.model.make_timestep()
        # Velocity should remain be [1, 0]
        assert np.allclose(self.model.get_ball_position,
                self.start + 2*np.array([1, 0]))

    def test_movement_2(self):
        """
        Base case: no walls, only y-dir vel.
        """
        self.model.set_acceleration(0, 2)
        self.model.make_timestep()
        # Velocity should now be [0, 2]
        
        assert np.allclose(self.model.get_ball_position,
                self.start + np.array([0, 2]))

        self.model.set_acceleration(0, 0)
        self.model.make_timestep()
        # Velocity should remain be [1, 0]
        assert np.allclose(self.model.get_ball_position,
                self.start + 2*np.array([0, 2]))

    def test_movement_3(self):
        """
        Base case: no walls, negative acc.
        """
        self.model.set_acceleration(1, 1)
        self.model.make_timestep()
        # Velocity should now be [1, 1]
        assert np.allclose(self.model.get_ball_position,
                self.start + np.array([1, 1]))

        self.model.set_acceleration(-2, -2)
        self.model.make_timestep()
        # Velocity should be [-1, -1]
        assert np.allclose(self.model.get_ball_position,
                self.start)
        self.model.make_timestep()
        assert np.allclose(self.model.get_ball_position,
                self.start + np.array([-1, -1]))

    def test_movement_4(self):
        """
        Base case: no walls, constant acc.
        """
        self.model.set_acceleration(1, 1)
        self.model.make_timestep()
        # Velocity should now be [1, 1]
        assert np.allclose(self.model.get_ball_position,
                self.start + np.array([1, 1]))

        self.model.make_timestep()
        # Velocity should be [2, 2]
        assert np.allclose(self.model.get_ball_position,
                self.start + np.array([3, 3]))
        # Velocity should be [3, 3]
        self.model.make_timestep()
        assert np.allclose(self.model.get_ball_position,
                self.start + np.array([6, 6]))

if (__name__ == "__main__"):
    unittest.main()
