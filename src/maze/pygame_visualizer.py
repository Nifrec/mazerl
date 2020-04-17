"""
Module that renders a maze simulation to a pygame Surface.

Author: Lulof Pirée
"""

# Library imports
import abc
from typing import Set, Iterable
from numbers import Number
from os import environ
# Disable pygame welcome message. Need to be set before 'import pygame'
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
# Local imports
from maze.record_types import Ball, Line, Size
from maze.abstract_visualizer import Visualizer

class PygameVisualizer(Visualizer):
    """
    Actual initialisable visualizer that uses pygame to create visualisations.
    """
    BACKGROUND_COLOR = pygame.Color(221, 251, 255)
    BALL_COLOR = pygame.Color( 255, 147, 0 )
    WALL_COLOR = pygame.Color(0, 228, 255)
    END_COLOR = pygame.Color(255, 0, 0)
    END_HALO_COLOR = pygame.Color(255, 135, 135)
    END_HALO_RAD = 15
    LINE_WIDTH = 1

    @staticmethod
    def render_ball(ball: Ball, target: pygame.Surface) \
            -> pygame.Surface:
        """
        Draws a Ball to a pygame.Surface instance,
        with PygameVisualizer.BALL_COLOR as color.
        """
        pygame.draw.circle(target, PygameVisualizer.BALL_COLOR, 
                ball.pos.round().astype(np.int), ball.rad, 0)
        return target

    @staticmethod
    def render_lines(lines: Set[Line], target: pygame.Surface) \
            -> pygame.Surface:
        """
        Draws a set of Line's to a pygame.Surface instance,
        with PygameVisualizer.WALL_COLOR as color,
        and width PygameVisualizer.LINE_WIDTH.
        """
        
        for line in lines:
            pygame.draw.line(target, PygameVisualizer.WALL_COLOR, line.p0,
                    line.p1, PygameVisualizer.LINE_WIDTH)

        return target

    @staticmethod
    def render_end(position: Iterable[Number], target: pygame.Surface) \
                -> pygame.Surface:
        """
        Render a maze endpoint to pygame.Surface.
        Draws the exact pixed in PygameVisualizer.END_COLOR,
        and surrounds it with a circle of color PygameVisualizer.END_HALO_COLOR
        and radius PygameVisualizer.END_HALO_RAD.
        """
        pygame.draw.circle(target, PygameVisualizer.END_HALO_COLOR, position,
                PygameVisualizer.END_HALO_RAD, 0)
        target.set_at(position, PygameVisualizer.END_COLOR)
        
        return target

    @staticmethod
    def create_rendered_object(size: Size) -> pygame.Surface:
        """
        Creates an empty Surface of the given size,
        filled with PygameVisualizer.BACKGROUND_COLOR.
        """
        output = pygame.Surface((size.x, size.y))
        output.fill(PygameVisualizer.BACKGROUND_COLOR)
        return output