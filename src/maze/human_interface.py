"""
File providing a human interface to the maze simulation.
This is not very polished, just to test the environment
and experience how the agent sees it.
Uses pygame for rendering and input.

Move the ball with the arrows or WASD, press ESC to quit.
If the goal is reached the screen becomes green for a second and resets,
if a wall is hit (i.e. you 'died'), the screen becomes red for a second 
and resets. Press F1 to generate a new .

Author: Lulof Pirée
"""
from numbers import Number
from os import environ
# Disable pygame welcome message. Need to be set before 'import pygame'
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
import time

from maze.model import Model, MazeLayout
from maze.record_types import Size, Line
from maze.pygame_visualizer import PygameVisualizer
from maze.pygame_ghost_visualizer import GhostVisualizer
from maze.maze_generator import MazeGenerator


FPS = 30
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480
FULLSCREEN = False
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
GAME_OVER_TIME = 1
BALL_RAD = 10
MAZE_OFFSET = 60
ACC_INCR_BUTTON_PRESS = 0.1


class HumanInterface():

    def __init__(self, fps: Number = FPS, 
            width: Number = WINDOW_WIDTH, height: Number = WINDOW_HEIGHT,
            fullscreen: bool=False):
        
        
        pygame.init()
        pygame.display.set_caption("MazeRL Human Interface")
        self.__size = Size(width, height)
        self.__maze_gen = MazeGenerator(self.__size, BALL_RAD, MAZE_OFFSET)
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
        self.__model = Model(self.__size, BALL_RAD)
        self.__visualizer = GhostVisualizer(self.__model)
        self.create_random_maze()
        self.render_all()
        self.run()

    def run(self):

        self.__is_running = True
        while(self.__is_running):
            self.__clock.tick(self.__fps)
            self.process_events()
            self.process_model()
            self.render_update_ball()
            pygame.display.flip()
        pygame.quit()

     
    def process_events(self):
        """
        This method registers keypresses and maps them to
        their effects.
        """
        # Now process pygame events (keyboard keys or controller buttons)
        for event in pygame.event.get():

            if (event.type == pygame.QUIT): # If user closes pygame window
                self.__is_running = False
            
            elif (event.type == pygame.KEYDOWN): # If a button is pressed
                if (event.key == pygame.K_ESCAPE): # User pressed ESC
                    self.__is_running = False

                if (event.key == pygame.K_F1):
                    self.create_random_maze()
        
        self.process_held_keys()

    def process_held_keys(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.__model.set_acceleration(-ACC_INCR_BUTTON_PRESS, 0)
        elif keys[pygame.K_w] or keys[pygame.K_UP]:
            self.__model.set_acceleration(0, -ACC_INCR_BUTTON_PRESS)
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.__model.set_acceleration(ACC_INCR_BUTTON_PRESS, 0)
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.__model.set_acceleration(0, ACC_INCR_BUTTON_PRESS)
                

    def process_model(self):
        """
        Updates the maze model and checks if the episde ended.
        """
        hit_wall = not self.__model.make_timestep()
        # Only accelerate while button is pressed, note that vel is preserved.
        self.__model.set_acceleration(0, 0)
        if (hit_wall):
            self.death_screen()
            self.create_random_maze()
            self.render_all()
        elif (self.__model.is_ball_at_finish()):
            self.win_screen()
            self.create_random_maze()
            self.render_all()

    def death_screen(self):
        """
        Show a red screen for a few seconds.
        """
        print("Hit a wall, negative end of episode reached.")
        self.__screen.fill(RED)
        pygame.display.flip()
        time.sleep(GAME_OVER_TIME)

    def win_screen(self):
        """
        Show a green screen for a few seconds.
        """
        print("Reached exit, positive end of episode reached.")
        self.__screen.fill(GREEN)
        pygame.display.flip()
        time.sleep(GAME_OVER_TIME)
    
    def render_all(self):
        self.__visualizer.render(self.__screen)

    def render_update_ball(self):
        self.__visualizer.update(self.__screen)

    def create_random_maze(self):
        self.__model.reset(self.__maze_gen.generate_maze())
        pygame.display.flip()

    def __create_test_maze(self):
        """
        Creates a simple hard-coded maze to test the rendering.
        """
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
                # Unrelated.
                Line(0, 200, 200, 200),
                Line(100, 250, 400, 250),
                Line(200, 400, 500, 400)
                ])

        start = np.array([300, 300])
        end = np.array([125, 20])
        layout = MazeLayout(lines, start, end, self.__size)

        self.__model.reset(layout)


if __name__ == "__main__":
    hi = HumanInterface(FPS, WINDOW_WIDTH, WINDOW_HEIGHT, FULLSCREEN)