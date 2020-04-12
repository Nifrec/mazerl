"""
Static methods used to create random maze layouts.

Author: Lulof Pirée
"""
# Library imports
from numbers import Number
from typing import Tuple, Set
import enum
# Local imports
from model import MazeLayout
from record_types import Line, Size

class Direction(enum.Enum):
    LEFT=1
    RIGHT=2
    UP=3
    DOWN=4

class MazeGenerator():

    @staticmethod
    def generate_maze(size: Size, radius: Number) -> MazeLayout:
        """
        Generate a new maze of the given size for a ball of the given
        radius (the radius needed to determine the minimum width of
        maze 'corridors').
        """
        pass

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

    def get_pos(self) -> Tuple[int]:
        return self.__x, self.__y