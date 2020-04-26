"""
Module that provides an interface for rendering a maze simulation.

Author: Lulof Pirée
"""
# Library imports
import abc
import pygame
import numpy as np
from typing import Any, Set, Iterable
from numbers import Number
# Local imports
from .record_types import Ball, Line, Size
#from .model import Model

class Visualizer(abc.ABC):
    """
    Abstract visualizer that provides interface for all visualizers.
    """
    @abc.abstractmethod
    def __init__(self, model):
        pass

    @abc.abstractmethod
    def render(self, screen: pygame.Surface):
        """
        Blits the whole maze to the screen. 
        """
        pass

    @abc.abstractmethod
    def update(self, screen: pygame.Surface):
        """
        Only updates portion of screen changed since last timestep.
        Assumes either update() or render() was called last timestep.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def render_ball(ball_rad: int, ball_pos: np.ndarray,
            target: pygame.Surface) -> Any:
        """
        Render a given Ball to a target picture object.
        """
        return target

    @staticmethod
    @abc.abstractmethod
    def render_lines(lines: Set[Line], target: Any) -> Any:
        """
        Render each Line in a set of lines to a target picture object.
        """
        return target

    @staticmethod
    @abc.abstractmethod
    def render_end(position: Iterable[Number], target: Any) -> Any:
        """
        Render a maze endpoint to a target picture object.
        """
        return target

    # @staticmethod
    # @abc.abstractmethod
    # def create_rendered_object(size: Size) -> Any:
    #     """
    #     Create an object that can render things.
    #     (e.g. a pygame.Surface when using pygame)
    #     """
    #     return None
