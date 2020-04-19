"""
Wrapper for the maze-environment that automatically converts observations
(screenshots) so 1-channel greyscale images.

Author: Lulof Pirée
"""
import numpy as np
from maze.environment_interface import Environment


class GreyScaleEnvironment(Environment):

    def __create_observation_array(self) -> np.ndarray:
        print('running overridden method')
        array = super()._Environment__create_observation_array()
        print(array.mean(axis=1))
        return array.mean(axis=1)

