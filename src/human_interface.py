"""
File providing a human interface to the maze simulation.
Uses pygame for rendering and input.

Author: Lulof Pirée
"""
from numbers import Number
from os import environ
# Disable pygame welcome message. Need to be set before 'import pygame'
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np

from model import Model, MazeLayout
from record_types import Size, Line
from pygame_visualizer import PygameVisualizer


FPS = 30
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480
FULLSCREEN = False



class HumanInterface():

    def __init__(self, fps: Number, 
            width: Number, height: Number, fullscreen: bool=False):
        pygame.init()
        pygame.display.set_caption("MazeRL Human Interface")
        self.__size = Size(width, height)
        self.__screen = pygame.display.set_mode((width, height))
        if fullscreen:
            pygame.display.toggle_fullscreen()
        self.__clock = pygame.time.Clock()
        # Frames/timesteps per second limit
        self.__fps = fps                                   
        self.__font = pygame.font.SysFont("DejaVu Sans", 20, bold = True)
        
        # Color background
        #self._screen.fill(backgroundcolor)
        pygame.display.flip()
        self.create_test_maze()
        self.run()

    def run(self):
        while(True):
            self.__clock.tick(self.__fps)
            pygame.display.flip()

    def create_test_maze(self):
        """
        Creates a simple hard-coded maze to test the rendering.
        """
        rad = 10
        model = Model(self.__size, rad)
        lines = set([
                # T
                Line(1, 1, 50, 1),
                Line(30, 1, 30, 100),
                # E
                Line(70, 1, 70, 100),
                Line(70, 1, 100, 1),
                Line(70, 40, 100, 40),
                Line(70, 100, 100, 100),
                # S
                Line(110, 1, 140, 1),
                Line(110, 1, 110, 40),
                Line(110, 40, 140, 40),
                Line(140, 40, 140, 100),
                Line(110, 100, 140, 100),
                # T
                Line(150, 1, 200, 1),
                Line(180, 1, 180, 100),
                ])

        print(self.__size.x, self.__size.y)
        start = np.array([300, 300])
        end = np.array([400, 400])
        layout = MazeLayout(lines, start, end, self.__size)

        model.reset(layout)
        surf = model.render(PygameVisualizer)
        self.__screen.blit(surf, (0, 0))


if __name__ == "__main__":
    hi = HumanInterface(FPS, WINDOW_WIDTH, WINDOW_HEIGHT, FULLSCREEN)