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

class BlockState(enum.Enum):
    # Neither in- nor out-direction set.
    EMPTY = 1
    # Either in- or out-direction set, but not both.
    PARIAL = 2
    # Both in- and out-directions have been configured.
    SET = 3


class MazeBlock():
    """
    Class used to generate small rectangular patches of the maze.

    Public fields: state: BlockState
            => EMPTY if neither in- nor out-direction set,
            => PARIAL if either in- or out-direction set, but not both.
            => SET if both in- and out-directions have been configured.
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
        self.state = BlockState.EMPTY

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

        if self.__direction_out is not None:
            self.state = BlockState.SET
        else:
            self.state = BlockState.PARIAL

    def set_direction_out(self, direction: Direction):
        """
        Sets the direction of edge though which the maze path leaves this block.
        NOTE: Does not update state.
        """
        assert direction != self.__direction_in, \
                "Cannot exit and enter via same edge"
        self.__direction_out = direction

        if self.__direction_in is not None:
            self.state = BlockState.SET
        else:
            self.state = BlockState.PARIAL        

    def reset(self):
        """
        Reset the stored in/out directions of the path through this block.
        """
        self.__direction_in = None
        self.__direction_out = None

    def get_center(self) -> np.ndarray:
        """
        Returns center coordinates as a 2-element numpy array.
        The coordinates are rounded down in case no single middle
        exists.
        """
        center_x = self.__x + math.floor((self.__width - 1) / 2)
        center_y = self.__y + math.floor((self.__height - 1) / 2)
        return np.array([center_x, center_y])

    def get_pos(self) -> Tuple[int]:
        return self.__x, self.__y

class MazeGrid():
    """
    Abstract data structure that represents a matrix of MazeBlocks.
    """

    def __init__(self, size: Size, block_size: Number):
        self.__blocks = self.__generate_block_partition(size, block_size)

    def __generate_block_partition(self, size: Size, block_size: Number) \
                -> np.ndarray:
        self.__num_rows = math.floor(size.x / block_size)
        self.__num_cols = math.floor(size.y / block_size)
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

    def __row_exists(self, row: Number):
        return (row >= 0) and (row < self.__num_rows)

    def __col_exists(self, col: Number):
        return (col >= 0) and (col < self.__num_col)

    def find_available_directions(self, row, col):
        """
        Given the indices of a block in self.__blocks, finds all Direction's
        such that the out direction of the given block can be set in that
        direction.
        (That is, a neighbour in that direction exist, and they are still
        in BlockState.EMPTY)
        """
        directions = set()
        if self.__row_exists(row - 1) \
                and (self.__blocks[row-1][col].state == BlockState.EMPTY):
            directions.add(Direction.UP)
        if self.__col_exists(col -1) \
                and (self.__blocks[row][col-1].state == BlockState.EMPTY):
            directions.add(Direction.LEFT)
        if self.__row_exists(row + 1) \
                and (self.__blocks[row+1][col].state == BlockState.EMPTY):
            directions.add(Direction.DOWN)
        if self.__col_exists(col+1) \
                and (self.__blocks[row][col+1].state == BlockState.EMPTY):
            directions.add(Direction.RIGHT)
        return directions

    def get_at(self, row: int, col: int) -> MazeBlock:
        return self.__blocks[row][col]

    def __reset_blocks(self):
        """
        Erases all in/out directions stored in all blocks.
        """
        for block in np.nditer(self.__blocks):
            block.reset()

    def choose_random_start(self, blocks: np.ndarray) -> MazeBlock:
        """
        Choses a random cell in the first column of a matrix of blocks,
        and returns the block of that cell.
        """
        row = random.randrange(self.__num_rows)
        col = 0
        block = self.__blocks[row][col]

        return block

    def choose_random_end(self, blocks: np.ndarray) -> MazeBlock:
        """
        Choses a random cell in the last column of a matrix of blocks,
        and returns the block of that cell.
        """
        row = random.randrange(self.__num_rows)
        col = self.__num_cols - 1
        block = self.__blocks[row][col]

        return block

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
        self.__size = size
        
    def generate_maze(self) -> MazeLayout:
        """
        Generate a new random maze for the configured settings.
        """
        start = self.__choose_random_start(self.__blocks)
        end = self.__choose_random_end(self.__blocks)
        current_block = start
        while (current_block != end):
            dirs = self.__find_available_directions(current_block)
            self.__set_random_directions(current_block, dirs)
            next_block = meh
        
        # walls = set()
        # for block in np.nditer(self.__blocks):
        #     block.generate_walls_where_no_path()
        
        # return MazeLayout(walls, start.get_center(), end.get_center(),
        #         self.__size)

        assert False, "TODO: finish and test!"

    # def __set_block_recursively(self, row: int, col: int) -> bool:
    #     current_block = self.__blocks[row][col]
    #     for direction in self.__find_available_directions():
    #         current_block.set_direction_out
    #         if not dead end:
    #             next_block.set_direction_in
    #             if recurse works:
    #                 return True
    #     return False