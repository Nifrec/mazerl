"""
Static methods used to create random maze layouts.

Author: Lulof Pirée
"""
# Library imports
from numbers import Number
from typing import Tuple
# Local imports
from model import MazeLayout
from record_types import Line, Size

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

    def __str__(self):
        output = f"MazeBlock({self.__x}, {self.__y}, {self.__width}" \
                + f",{self.__height})"
        return output

    def generate_wall_left(self) -> Line:
        """
        Generate a Line along the left edge of this block.
        (Located just _inside_ the block)
        """
        return Line(self.__x, self.__y, self.__x, self.__y + self.__height)

    def generate_wall_right(self) -> Line:
        """
        Generate a Line along the right edge of this block.
        (Located just _inside_ the block)
        """
        return Line(self.__x + self.__width, self.__y,
                self.__x + self.__width, self.__y + self.__height)

    def generate_wall_top(self) -> Line:
        """
        Generate a Line along the top edge of this block.
        (Located just _inside_ the block)
        """
        return Line(self.__x , self.__y, self.__x + self.__width, self.__y)

    def generate_wall_bottom(self) -> Line:
        """
        Generate a Line along the bottom edge of this block.
        (Located just _inside_ the block)
        """
        return Line(self.__x , self.__y + self.__height,
                self.__x + self.__width, self.__y + self.__height)

    def get_pos(self) -> Tuple[int]:
        return self.__x, self.__y