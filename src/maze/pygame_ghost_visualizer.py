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
        self.__is_first_render = True

    def render(self, screen: pygame.Surface):
        super().render(screen)

    def _draw_below_ball_effects_hook(self, screen: pygame.Surface):
        """
        Adds cleaning up of ghost-ball dirty rect on top
        of PygameVisualizer's update()
        """
        ball_pos = self._model.get_ball_position()
        ball_rad = self._model.get_ball_rad()
        if ( self.__is_first_render):
            # Velocity and acceleration are not needed, only the size 
            # and position.
            self.__ghost_ball = Ball(ball_pos[0], ball_pos[1], ball_rad)
            self.__ghost_ball.pos = self.__ghost_ball.pos.round().astype(np.int) - 1
            self.__is_first_render = False

        dirty_rect_size = (2*self.__ghost_ball.rad, 2*self.__ghost_ball.rad)
        dirty_rect = pygame.Rect(self.__ghost_ball.pos, dirty_rect_size)
        screen.blit(self._layout_surf, 
                self.__ghost_ball.pos - self.__ghost_ball.rad,
                dirty_rect)

        self.__ghost_ball.pos = GhostVisualizer.GHOST_MOVEMENT_DELAY \
            * self.__ghost_ball.pos \
            + GhostVisualizer.GHOST_MOVEMENT_DELAY * ball_pos
        self.__ghost_ball.pos = self.__ghost_ball.pos.round().astype(np.int)
        
        pygame.draw.circle(screen, GhostVisualizer.GHOST_COLOR,
                self.__ghost_ball.pos,
                self.__ghost_ball.rad, 0)