"""
File providing a gym-like interface to the MazeRL game for
AI agents.

Author: Lulof Pirée
"""
# Library imports
from numbers import Number
from typing import Tuple, Iterable
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

# Standard reward per non-final timestep.
REWARD_STEP = -1
# Reward for successfully reaching the exit of the maze.
REWARD_WIN = 200
# Reward for hitting a wall or window boundary (terminates episode).
REWARD_DEATH = -100

class Environment():

    def __init__(self, size=Size(WINDOW_WIDTH, WINDOW_HEIGHT)):
        self.__size = size
        self.__model = Model(size, BALL_RAD)
        self.__generator = MazeGenerator(size, BALL_RAD, MAZE_OFFSET)

        self.__init_display()

    def step(self, action: Iterable) -> Tuple[np.ndarray, Number, bool]:
        """
        Takes a chosen action and computes the next time-step of the
        simulation.

        Arguments:
        * action: iterable of 2 floats, 
                chosen x- and y-acceleration respectively.

        Returns:
            * observation: 3D numpy int-type ndarray, 
                    RGB representation of reached state (screenshot).
                    (shape: channel(=3) x width x heigth)
            * reward: Number, rewards obtained for the given action.
            * done: bool, whether the reached state is the final state of
                    an episode.
        """
        if (len(action) != 2):
            raise ValueError(self.__class__.__name__ 
                + "step(): expected exactly two numbers in action")
        
        self.__model.set_acceleration(action[0], action[1])
        died = not self.__model.make_timestep()
        won = self.__model.is_ball_at_finish()
        reward = self.__compute_reward(died, won)
        done = died or won
        return self.__create_observation_array(), reward, done

    def __compute_reward(self, died: bool, won: bool) -> Number:
        if died:
            return REWARD_DEATH
        elif won:
            return REWARD_WIN
        else:
            return REWARD_STEP

    def __create_observation_array(self) -> np.ndarray:
        self.__update_display()
        red = pygame.surfarray.pixels_red(self.__screen)
        green = pygame.surfarray.pixels_green(self.__screen)
        blue = pygame.surfarray.pixels_blue(self.__screen)
        return np.array([red, green, blue])

    def reset(self) -> np.ndarray:
        """
        Generates a new maze and sets the ball at the starting position with
        0 velocity.
        
        Returns:
            * observation: 3D numpy int-type ndarray, 
                    RGB representation of reached state (screenshot).
                    (shape: channel(=3) x width x heigth)
        """
        self.__model.reset(self.__generator.generate_maze())
        return self.__create_observation_array()

    def render(self):
        pygame.display.flip()

    def __init_display(self):
        """
        Launch the pygame window.
        """
        pygame.init()
        pygame.display.set_caption("MazeRL AI Interface")
        self.__screen = pygame.display.set_mode((self.__size.x, self.__size.y))

    def __update_display(self):
        """
        Updates the stored pygame surface according to the new
        state of the maze, but does not flip
        what is actually displayed on the screen on its own.
        (Use self.render() or pygame.display.flip() for that last step)
        """
        surf = self.__model.render(PygameVisualizer)
        self.__screen.blit(surf, (0, 0))
