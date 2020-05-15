"""
Model providing the maze-simulation of the environment.

Author: Lulof Pirée
"""
# Library imports
import math
from numbers import Number
from typing import Any, Set
import numpy as np
import enum
# Local imports
from src.maze.distances import euclidean_dist, dist_line_to_line, \
        dist_point_line_segment,is_ball_in_rect
from src.maze.record_types import Size, Line, Ball
from src.maze.abstract_visualizer import Visualizer

class MazeLayout():
    """
    Class representing a maze room: defines the walls,
    the starting position and the end position (the 'exit').
    """

    def __init__(self, lines: set, start_point: np.ndarray,
            end_point: np.ndarray, size: Size):
        """
        Arguments:
        * lines: set of Line objects, walls of the .
        * start_point: [x, y] array, starting coordinates of player in .
        * end_point: [x, y] array, exit coordinates of maze (i.e. the player's
                goal)

        NOTE: does not check if the start or end coincides with a wall.
                (They should definitely not.)
        """
        self.__lines = lines
        self.__start = start_point
        self.__end = end_point
        self.__size = size

        # Some of these checks use the fields initialized above.
        assert(isinstance(size, Size))
        self.__check_validity_lines(lines)
        check_validity_point(start_point)
        self.__check_validity_location_point(start_point)
        check_validity_point(end_point)
        self.__check_validity_location_point(end_point)
        

    def __check_validity_lines(self, lines):
        """
        Checks if all the line endpoints are within the Size
        of this MazeLayout. Not that, because indices start at 0,
        a line's coordinates must be strictly less than the maxima of the
        size.
        Returns:
        * Boolean: True if all lines in the rectange defined by (0, 0)
                and self.__size.
        """
        assert(isinstance(lines, set))
        for line in lines:
            assert(isinstance(line, Line))
            for point in (line.p0, line.p1):
                self.__check_validity_location_point(point)

    def __check_validity_location_point(self, point: np.ndarray):
        if (point[0] < 0) \
                or (point[1] < 0) \
                or (point[0] >= self.__size.x) \
                or (point[1] >= self.__size.y):
            raise ValueError(self.__class__.__name__ 
                + f": Invalid input: datapoint {point} beyond layout size.")

    def is_ball_at_finish(self, ball: Ball):
        return (euclidean_dist(self.__end, ball.pos) <= ball.rad)

    def does_ball_hit_wall(self, ball):
        assert(isinstance(ball, Ball))

        for line in self.__lines:
            if (self.__collides(ball, line)):
                return True
        return False

    def compute_min_dist_line_to_wall(self, line: Line) -> float:
        """
        Returns the minimum distance of a given line to any of the walls
        in this MazeLayout.
        NOTE: size borders are not considered walls here.
        """
        min_dist = float("inf")
        for maze_line in self.__lines:
             min_dist = min(min_dist, dist_line_to_line(maze_line, line))
        return min_dist

    def __collides(self, ball: Ball, line: Line):
        """
        Checks if a given ball collides with a given line.
        Note that just hitting without overlapping counts as hitting.
        Also note that lines have a finite length here.
        """
        distance = dist_point_line_segment(ball.pos, line)
        
        if (distance <= ball.rad):
            return True
        else:
            return False

    def get_start(self) -> np.ndarray:
        return self.__start.copy()
    
    def get_end(self) -> np.ndarray:
        return self.__end.copy()

    def get_size(self) -> Size:
        return self.__size.copy()

    def get_lines(self) -> Set[Line]:
        """
        Returns a copy of all the lines stored in this MazeLayout.
        """
        output = set()
        for line in self.__lines:
            output.add(line.copy())
        return output

class Model():

    def __init__(self, size: Size, ball_rad: Number):
        assert(isinstance(size, Size))
        self.__size = size
        self.__ball_rad = ball_rad
        self.__ball = Ball(0, 0, self.__ball_rad)
        self.__layout = None

    def set_acceleration(self, x_acc: Number, y_acc: Number):
        self.__ball.acc = np.array([x_acc, y_acc], dtype=np.float)

    def reset(self, new_layout: MazeLayout = None):
        """
        Resets the ball position to the start of the current
        MazeLayout. Moreover, changes the layout if a new
        layout is supplied. The first time reset() is called,
        a new layout must always be supplied.
        Arguments:
            * new_layout: MazeLayout, new room of maze to load.
            If None, the old layout will re resued (or an error
            thrown if no layout exists)
        """
        self.__check_valid_new_layout(new_layout)
        if new_layout is not None:
            assert(isinstance(new_layout, MazeLayout))
            self.__layout = new_layout

        self.__ball.acc = np.array([0, 0], dtype=np.float)
        self.__ball.vel = np.array([0, 0], dtype=np.float)
        self.__ball.pos = self.__layout.get_start().astype(np.float)

    def __check_valid_new_layout(self, new_layout: MazeLayout):
        if (new_layout is None) and (self.__layout is None):
            raise RuntimeError(self.__class__.__name__ 
                + "reset(): no new_layout given and cannot reuse "
                + "non-existing layout")

        new_layout_size = new_layout.get_size()
        if (new_layout_size.x != self.__size.x) or \
                (new_layout_size.y != self.__size.y):
                raise ValueError(self.__class__.__name__
                        + f"reset(): new_layout has wrong size, {self.__size}"
                        + "expected")

    def is_ball_at_finish(self) -> bool:
        return self.__layout.is_ball_at_finish(self.__ball)

    def does_ball_hit_wall(self) -> bool:
        output = self.__layout.does_ball_hit_wall(self.__ball)
        output = output or self.__does_ball_hit_boundary()
        return output

    def __does_ball_hit_boundary(self) -> bool:
        return (is_ball_in_rect(self.__ball, self.__size) == False)

    def get_ball_position(self) -> np.ndarray:
        return self.__ball.pos.copy()

    def make_timestep(self) -> bool:
        """
        Updates velocity and position ball.
        Returns True if the movement was valid (i.e. no walls or boundaries
        were hit), False otherwise.
        """
        self.__ball.vel += self.__ball.acc
        old_pos = self.__ball.pos.copy()
        self.__ball.pos += self.__ball.vel

        movement_line = Line(old_pos[0], old_pos[1], 
                self.__ball.pos[0], self.__ball.pos[1])
        # Collision during movement
        if (self.__layout.compute_min_dist_line_to_wall(movement_line) 
                <= self.__ball.rad):
            return False
        # Collision with borders (and walls at new pos, but was already checked)
        elif self.does_ball_hit_wall():
            return False
        else:
            return True

    def get_layout(self):
        return self.__layout

    def get_ball_rad(self):
        return self.__ball.rad

def check_validity_point(point: np.ndarray):
    if not isinstance(point, np.ndarray):
        raise ValueError("Invalid input point, np.ndarray expected.")
    if (len(point) != 2):
        raise ValueError("Invalid input point, length 2 expected.")
