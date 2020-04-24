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

class GhostVisualizer(PygameVisualizer):
    # Set alpha to 50% transparency.
    # GHOST_COLOR = pygame.Color(
    #     PygameVisualizer.BALL_COLOR.r,
    #     PygameVisualizer.BALL_COLOR.g,
    #     PygameVisualizer.BALL_COLOR.b,
    #     128
    # )
    GHOST_COLOR = pygame.Color(128, 0, 0, 10)

    def __init__(self):
        self.__prev_ball = None

    def render_ball(self, ball: Ball, target: pygame.Surface) \
            -> pygame.Surface:
        """
        Draws a Ball to a pygame.Surface instance,
        with PygameVisualizer.BALL_COLOR as color.
        Also draws a transparent 'ghost ball' at the 
        previous location of the ball.

        NOTE: not static like in PygameVisualizer because it has internal state.
        (i.e. the previous location of the ball.)
        """
        if (self.__prev_ball == None):
            # Velocity and acceleration are not needed, only the size 
            # and position.
            self.__prev_ball = Ball(ball.pos[0], ball.pos[1], ball.rad)
        transparent_surf = pygame.Surface(target.get_size(), flags=pygame.SRCALPHA)
        pygame.draw.circle(transparent_surf, PygameVisualizer.BALL_COLOR,
                self.__prev_ball.pos.round().astype(np.int),
                self.__prev_ball.rad, 0)
        transparent_surf.set_alpha(128)
        target.blit(transparent_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        self.__prev_ball.pos = 0.5 * self.__prev_ball.pos + 0.5*ball.pos.copy()

        target = super().render_ball(ball, target)
        
        return target

    @staticmethod
    def create_rendered_object(size: Size) -> pygame.Surface:
        """
        Same as from super, but output Surface also has alpha enabled.
        """
        output = pygame.Surface((size.x, size.y), flags=pygame.SRCALPHA)
        output.fill(PygameVisualizer.BACKGROUND_COLOR)
        return output