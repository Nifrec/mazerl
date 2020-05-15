"""
Wrapper for the maze-environment that automatically converts observations
(screenshots) so 1-channel greyscale images.

Author: Lulof Pirée
"""
import numpy as np
from typing import Iterable, Tuple
from numbers import Number
from src.maze.environment_interface import Environment, WINDOW_HEIGHT, WINDOW_WIDTH


class GreyScaleEnvironment(Environment):

    def create_observation_array(self) -> np.ndarray:
        array = super().create_observation_array()
        return array.mean(axis=0).reshape(1, WINDOW_WIDTH, WINDOW_HEIGHT)

    def step(self, action: Iterable) -> Tuple[np.ndarray, Number, bool, None]:
        """
        Same as environment_interface.Environment.step(), but with added
        None in output (to have same output size as gym's environments)
        """
        return super().step(action) + (None,)

