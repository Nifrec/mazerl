"""
File providing a gym-like interface to the MazeRL game for
AI agents.

Author: Lulof Pirée
"""
# Library imports
from numbers import Number
from typing import Tuple
from os import environ
# Disable pygame welcome message. Need to be set before 'import pygame'
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
import time

# Local imports
from model import Model
from record_types import Size
from pygame_visualizer import PygameVisualizer
from maze_generator import MazeGenerator

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480
BALL_RAD = 10
MAZE_OFFSET = 60

class Environment():

    def __init__(self, size=Size(WINDOW_WIDTH, WINDOW_HEIGHT)):
        self.__display_is_initialized = False
        self.__size = size
        self.__model = Model(size, BALL_RAD)
        self.__generator = MazeGenerator(size, BALL_RAD, MAZE_OFFSET)

    def step(self, action: np.ndarray) -> Tuple:
        if (len(action) != 2):
            raise ValueError(self.__class__.__name__ 
                + "step(): expected exactly two numbers in action")
        
        self.__model.set_acceleration(action[0], action[1])
        died = self.__model.make_timestep()

    def reset(self) -> np.ndarray:
        self.__model.reset()

    def render(self):
        if not self.__display_is_initialized:
            self.__init_display()

        surf = self.__model.render(PygameVisualizer)
        self.__screen.blit(surf, (0, 0))
        pygame.display.flip()

    def __init_display(self):
        """
        Launch the pygame window.
        """
        pygame.init()
        pygame.display.set_caption("MazeRL AI Interface")
        self.__screen = pygame.display.set_mode((self.__size.x, self.__size.y))
