"""
Extension to the PygameVisualizer that also
renders the previous location of the ball as transparent.
This gives an idea of motion when given only a single screenshot.

Author: Lulof Pirée
"""
import pygame
import numpy as np

from maze.record_types import Ball, Size
from maze.pygame_visualizer import PygameVisualizer
from .model import Model

class GhostVisualizer(PygameVisualizer):
    
    GHOST_COLOR = pygame.Color(128, 0, 0, 10)
    GHOST_MOVEMENT_DELAY = 0.5

    def __init__(self, model: Model):
        super().__init__(model)
        self.__prev_ball = None

    def render(self, screen: pygame.Surface):
        super().render(screen)

    def render_ball(self, ball: Ball, target: pygame.Surface) \
            -> pygame.Surface:
        """
        Draws a Ball to a pygame.Surface instance,
        with PygameVisualizer.BALL_COLOR as color.
        Also draws a transparent 'ghost ball' that lags behind 
        the location of the ball.

        NOTE: not static like in PygameVisualizer because it has internal state.
        (i.e. the previous location of the ball.)
        """
        if (self.__prev_ball == None):
            # Velocity and acceleration are not needed, only the size 
            # and position.
            self.__prev_ball = Ball(ball.pos[0], ball.pos[1], ball.rad)

        pygame.draw.circle(target, GhostVisualizer.GHOST_COLOR,
                self.__prev_ball.pos.round().astype(np.int),
                self.__prev_ball.rad, 0)
        self.__prev_ball.pos = GhostVisualizer.GHOST_MOVEMENT_DELAY \
            * self.__prev_ball.pos \
            + GhostVisualizer.GHOST_MOVEMENT_DELAY * ball.pos.copy()

        target = super().render_ball(ball, target)
        
        return target