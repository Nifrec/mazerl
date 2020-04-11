"""
Module that renders a maze simulation to a pygame Surface.

Author: Lulof Pirée
"""

# Library imports
import abc
from typing import Set
from os import environ
# Disable pygame welcome message. Need to be set before 'import pygame'
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
# Local imports
from record_types import Ball, Line, Size
from abstract_visualizer import Visualizer

class PygameVisualizer(Visualizer):
    """
    Actual initialisable visualizer that uses pygame to create visualisations.
    """
    BACKGROUND_COLOR = pygame.Color(221, 251, 255)
    BALL_COLOR = pygame.Color( 255, 147, 0 )
    WALL_COLOR = pygame.Color(0, 228, 255)
    LINE_WIDTH = 1

    @staticmethod
    def render_ball(ball: Ball, target: pygame.Surface) \
            -> pygame.Surface:
        """
        Draws a Ball to a pygame.Surface instance,
        with PygameVisualizer.BALL_COLOR as color.
        """
        pygame.draw.circle(target, PygameVisualizer.BALL_COLOR, ball.pos,
                ball.rad, 0)
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
    def create_rendered_object(size: Size) -> pygame.Surface:
        """
        Creates an empty Surface of the given size,
        filled with PygameVisualizer.BACKGROUND_COLOR.
        """
        output = pygame.Surface((size.x, size.y))
        output.fill(PygameVisualizer.BACKGROUND_COLOR)
        return output