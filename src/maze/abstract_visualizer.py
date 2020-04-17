"""
Module that provides an interface for rendering a maze simulation.

Author: Lulof Pirée
"""
# Library imports
import abc
from typing import Any, Set, Iterable
from numbers import Number
# Local imports
from maze.record_types import Ball, Line, Size

class Visualizer(abc.ABC):
    """
    Abstract visualizer that provides interface for all visualizers.
    """

    @staticmethod
    @abc.abstractmethod
    def render_ball(ball: Ball, target: Any) -> Any:
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

    @staticmethod
    @abc.abstractmethod
    def create_rendered_object(size: Size) -> Any:
        """
        Create an object that can render things.
        (e.g. a pygame.Surface when using pygame)
        """
        return None
