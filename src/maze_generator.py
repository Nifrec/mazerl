"""
Static methods used to create random maze layouts.

Author: Lulof Pirée
"""
# Library imports
from numbers import Number
from typing import Tuple, Set
import enum
import math
import numpy as np
import random
# Local imports
from model import MazeLayout
from record_types import Line, Size

class Direction(enum.Enum):
    LEFT=1
    RIGHT=2
    UP=3
    DOWN=4


class MazeBlock():
    """
    Class used to generate small rectangular patches of the maze.
    """

    def __init__(self, x: Number, y: Number, width: Number, height: Number):
        """
        Sets the region of the maze that this block covers.
        (i.e. specifies a rectangle.)

        Arguments:
        * x, y: position of left-upper corner of MazeBlock in maze.
        * width, height: size of this MazeBlock.
        """
        self.__x = x
        self.__y = y
        self.__width = width
        self.__height = height
        self.__direction_in = None
        self.__direction_out = None

    def __str__(self):
        output = f"MazeBlock({self.__x}, {self.__y}, {self.__width}" \
                + f",{self.__height})"
        return output

    def set_up(self, in_direction: Direction, out_direction: Direction) \
                -> Set[Line]:
        """
        Facade method that sets the directions of the path though
        this MazeBlock and generates walls (Lines) for the other edges.
        """
        self.set_direction_in(in_direction)
        self.set_direction_out(out_direction)
        return self.generate_walls_where_no_path()

    def generate_walls_where_no_path(self) -> Set[Line]:
        """
        Create set of walls that block the edges though which the maze
        path does not traverse. If the in and/or out direction is not set,
        then simply more walls will be generated.
        """
        walls = set()
        for direction in Direction:
            if (direction != self.__direction_in) \
                    and (direction != self.__direction_out):
                walls.add(self.generate_wall(direction))
        return walls

    def generate_wall(self, direction) -> Line:
        """
        Generate a Line along the left, right, top or bottom edge of this block.
        (Located just _inside_ the block)
        """
        if (direction == Direction.LEFT):
            return Line(self.__x, self.__y, self.__x, self.__y + self.__height)
        elif (direction == direction.RIGHT):
            return Line(self.__x + self.__width, self.__y,
                self.__x + self.__width, self.__y + self.__height)
        elif (direction == direction.UP):
            return Line(self.__x , self.__y, self.__x + self.__width, self.__y)
        elif (direction == direction.DOWN):
            return Line(self.__x , self.__y + self.__height,
                self.__x + self.__width, self.__y + self.__height)
        else:
            raise ValueError(self.__class__.__name__ + "generate_wall():"
                    + f"invalid direction '{direction}'")

    def set_direction_in(self, direction: Direction):
        """
        Sets the direction of edge though which the maze path enters this block.
        """
        assert direction != self.__direction_out, \
                "Cannot exit and enter via same edge"
        self.__direction_in = direction

    def set_direction_out(self, direction: Direction):
        """
        Sets the direction of edge though which the maze path leaves this block.
        """
        assert direction != self.__direction_in, \
                "Cannot exit and enter via same edge"
        self.__direction_out = direction

    def reset(self):
        """
        Reset the stored in/out directions of the path through this block.
        """
        self.__direction_in = None
        self.__direction_out = None

    def get_pos(self) -> Tuple[int]:
        return self.__x, self.__y

class MazeGenerator():

    def __init__(self, size: Size, radius: Number, offset: Number):
        """
        Configure the settings for generating random mazes.

        Arguments:
        * size: target size of generated MazeLayout.
        * radius: radius of ball for which the maze is intended
                (needed to determine the minimum width of
                maze 'corridors')
        * offset: minimum extra space added to both sides of a corridor
                to give the ball more space. 
                This is a measure for difficulty, as narrow corridors are harder
                to navigate through.
        """
        # +2 because each on-path block may contain 2 1-pixel-broad walls.
        block_size = radius + 2 + offset
        self.__blocks = self.__generate_block_partition(size, block_size)
        
    def generate_maze(self) -> MazeLayout:
        """
        Generate a new random maze for the configured settings.
        """
        pass
        
    def __generate_block_partition(self, size: Size, block_size: Number) \
                -> np.ndarray:
        self.__num_rows = math.ceil(size.x / block_size)
        self.__num_cols = math.ceil(size.y / block_size)
        blocks = np.empty((self.__num_rows, self.__num_cols), dtype=MazeBlock)
        x = 0
        y = 0
        for row in range(self.__num_rows):
            y = 0
            for col in range(self.__num_cols):
                blocks[row][col] = MazeBlock(x, y, block_size, block_size)
                y += block_size
            x += block_size
        return blocks

    def __reset_blocks(self):
        """
        Erases all in/out directions stored in all blocks.
        """
        for block in np.nditer(self.__blocks):
            block.reset()

    def __choose_random_start(self, blocks: np.ndarray) -> np.ndarray:
        """
        Choses a random cell in the first column of a matrix of blocks,
        and returns the indices of that block.
        """
        row = random.randrange(self.__num_rows)
        return np.array([row, 0])

    def __choose_random_end(self, blocks: np.ndarray) -> np.ndarray:
        """
        Choses a random cell in the last column of a matrix of blocks,
        and returns the indices of that block.
        """
        row = random.randrange(self.__num_rows)
        return np.array([row, self.__num_cols - 1])