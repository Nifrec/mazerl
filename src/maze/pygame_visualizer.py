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
from .model import Model

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

    def __init__(self, model: Model):
        self.__model = model
        self.__is_first_render_call = True

    def render(self, screen: pygame.Surface):
        """
        Updates the display with changes since the last render() call.
        The first time render() is called it will draw the entire maze.
        
        Note that this assumes no other function or method blits anything
        on the display inbetween render() calls.
        """
        self.__old_ball_pos: np.ndarray
        ball_pos = self.__model.get_ball_position() // np.ndarray
        ball_rad = self.__model.get_ball_rad()
        if (self.__is_first_render_call):
            self.__is_first_render_call = False
            self.__render_maze_layout()
        else:
            dirty_rect_size = (ball_rad, ball_rad)
            dirty_rect = pygame.Rect(self.__old_ball_pos, dirty_rect_size)
            display.blit(display, self.__old_ball_pos, dirty_rect)

        # Pos of right-upper corner of rectangle surrounding the ball.
        self.__old_ball_pos =  (ball_pos - ball_rad).round().astype(np.int)
        
        self.render_ball(ball_rad, ball_pos, display)
    
    def __render_maze_layout(self):
        self.__layout_surf = pygame.Surface(display.get_size())
        self.__layout_surf.fill(self.BACKGROUND_COLOR)
        self.render_end(self.__model.get_layout().get_end(),
                self.__layout_surf)
        self.render_lines(self.__model.get_layout().get_lines(),
                self.__layout_surf)
        display.blit(self.__layout_surf, (0, 0))

    @staticmethod
    def render_ball(ball_rad: int, ball_pos: np.ndarray,
            target: pygame.Surface) -> pygame.Surface:
        """
        Draws a Ball to a pygame.Surface instance,
        with PygameVisualizer.BALL_COLOR as color.
        """
        pygame.draw.circle(target, PygameVisualizer.BALL_COLOR, 
                ball_pos.round().astype(np.int), ball_rad, 0)
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

    # def create_rendered_object(size: Size) -> pygame.Surface:
    #     """
    #     Creates an empty Surface of the given size,
    #     filled with PygameVisualizer.BACKGROUND_COLOR.
    #     """
    #     output = pygame.Surface((size.x, size.y))
    #     output.fill(PygameVisualizer.BACKGROUND_COLOR)
    #     return output