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
        # Velocity and acceleration are not needed, only the size 
        # and position.
        pos = model.get_ball_position()
        rad = model.get_ball_rad()
        self.__ghost_ball = Ball(pos[0], pos[1], rad)
        self.__ghost_ball.pos = self.__ghost_ball.pos.round().astype(np.int)

    def render(self, screen: pygame.Surface):
        super().render(screen)

    def update(self, screen: pygame.Surface):
        """
        Adds cleaning up of ghost-ball dirty rect on top
        of PygameVisualizer's update()
        """
        dirty_rect_size = (2*self.__ghost_ball.rad, 2*self.__ghost_ball.rad)
        dirty_rect = pygame.Rect(self.__ghost_ball.pos, dirty_rect_size)
        screen.blit(self._layout_surf, 
                self.__ghost_ball.pos - ball_rad,
                dirty_rect)
        super().update(screen)

    def render_ball(self, ball_rad: int, ball_pos: np.ndarray,
            target: pygame.Surface) -> pygame.Surface:
        
        pygame.draw.circle(target, GhostVisualizer.GHOST_COLOR,
                self.__ghost_ball.pos,
                self.__ghost_ball.rad, 0)
        self.__ghost_ball.pos = GhostVisualizer.GHOST_MOVEMENT_DELAY \
            * self.__ghost_ball.pos \
            + GhostVisualizer.GHOST_MOVEMENT_DELAY * ball.pos.copy()
        self.__ghost_ball.pos = self.__ghost_ball.pos.round().astype(np.int)

        target = super().render_ball(ball_rad, ball_pos, target)
        
        return target