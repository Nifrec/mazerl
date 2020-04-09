"""
Model providing the maze-simulation of the environment.

Author: Lulof Pirée
"""
import math
import numpy as np
from numbers import Number
from numpy import linalg as LA
import enum

class Orientation(enum.Enum):
    CLOCKWISE = 1
    COUNTERCLOCKWISE = 2
    COLLINEAR = 3


class Size():
    """
    Record type for width & height of rectangular coordinate planes.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def copy(self):
        return Size(self.x, self.y)

class Line():
    """
    Record type for a (finite) straight line segment 
    in Cartesian coordinate space.
    """

    def __init__(self, x1, y1, x2, y2):
        self.p0 = np.array([x1, y1])
        self.p1 = np.array([x2, y2])
        #self.length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # Coefficients of equation ax + by + c = 0 for this line.
        

    def __str__(self):
        return f"Line ({self.p0}, {self.p1})"

class Ball():
    """
    Record type for a moving ball, has a position, radius,
    velocity (in 2 dimensions) and acceleration (in 2 dimensions).
    """
    def __init__(self, x, y, rad, x_vel=0, y_vel=0, x_acc=0, y_acc=0):
        self.pos = np.array([x, y])
        self.rad = rad
        self.vel = np.array([x_vel, y_vel])
        self.acc = np.array([x_acc, y_acc])

class MazeLayout():
    """
    Class representing a maze room: defines the walls,
    the starting position and the end position (the 'exit').
    """

    def __init__(self, lines, start_point, end_point, size):
        """
        Arguments:
        * lines: set of Line objects, walls of the maze.
        * start_point: [x, y] array, starting coordinates of player in maze.
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
                + ": Invalid input: data beyond layout size.")

    def is_ball_at_finish(self, ball: Ball):
        return (euclidean_dist(self.__end, ball.pos) <= ball.rad)

    def does_ball_hit_wall(self, ball):
        assert(isinstance(ball, Ball))

        for line in self.__lines:
            if (self.__collides(ball, line)):
                return True
        return False

    def __collides(self, ball: Ball, line: Line):
        """
        Checks if a given ball collides with a given line.
        Note that just hitting without overlapping counts as hitting.
        Also note that lines have a finite length here.
        """
        distance = dist_point_segment_line(ball.pos, line)
        
        if (distance <= ball.rad):
            return True
        else:
            return False

    def get_start(self):
        return self.__start.copy()
    
    def get_end(self):
        return self.__end.copy()

    def get_size(self):
        return self.__size.copy()

class Model():

    def __init__(self, size: Size, ball_rad: Number):
        assert(isinstance(size, Size))
        self.__size = size
        self.__ball_rad = ball_rad
        self.__ball = Ball(0, 0, self.__ball_rad)
        self.__layout = None

    def set_acceleration(self, x_acc: Number, y_acc: Number):
        self.__ball.acc = np.array([x_acc, y_acc])

    def reset(self, new_layout = None):
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

        self.__ball.acc = np.array([0, 0])
        self.__ball.vel = np.array([0, 0])
        self.__ball.pos = self.__layout.get_start()

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

    def is_ball_at_finish(self):
        return self.__layout.is_ball_at_finish(self.__ball)

    def does_ball_hit_wall(self):
        output = self.__layout.does_ball_hit_wall(self.__ball)
        output = output or self.__does_ball_hit_boundary()
        return output

    def __does_ball_hit_boundary(self):
        return (is_ball_in_rect(self.__ball, self.__size) == False)

    def get_ball_position(self):
        return self.__ball.pos.copy()

    def make_timestep(self):
        self.__ball.vel += self.__ball.acc
        self.__ball.pos += self.__ball.vel

    def render(self):
        pass


def get_line_implicit_coefs(line):
    """
    Returns the coeficients of the implicit 2D equation
    of a line.
    That it, it returns a, b, c in :
            ax + by + c = 0
    """
    # Proof:
    # [x2]   [x1-x2]          [x]
    # [y2] + [y1-y2]*lambda = [y]
    # 
    # --> lambda = (x - x2)/(x1-x2)
    # --> y = y2 + (y1-y2)*lambda = y2 + (y1-y2)(x - x2)/(x1-x2)
    # --> y(x1 - x2) + x(y2 - y1) + y2(x2 - x1) +x2(y1 - y2) = 0
    x1 = line.p0[0]
    y1 = line.p0[1]
    x2 = line.p1[0]
    y2 = line.p1[1]
    
    a = y2 - y1
    b = x1 - x2
    c = y2*(x2 - x1) + x2*(y1 - y2)

    return a, b, c

def dist_line_segment_line_segment(line_0: Line, line_1: Line) -> Number:
    """
    Computes the shortest distance between two line segments.
    """
    # 2 cases: the lines intersect -> dist = 0.
    # or they do not, and the dist is dist(endpoint_one_line, other_line).
    # In the latter scenario there are only 4 combinations to check.
    if (do_lines_intersect(line_0, line_1)):
        return 0
    else:
        assert False, "TODO"


def do_lines_intersect(line_0: Line, line_1: Line) -> bool:
    """
    Returns whether two finite line segments intersect.
    """
    # Theory:
    # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    # https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    
    # Compute orientations of all triangles created by taking one line
    # and one endpoint of the other line.
    # For lines AB and XY, that are O(A, B, X), O(A, B, Y),
    # O(X, Y, A), O(X, Y, B) (where O is the orientation)
    orient_0 = compute_orientation_points(line_0.p0, line_0.p1, line_1.p0)
    orient_1 = compute_orientation_points(line_0.p0, line_0.p1, line_1.p1)
    orient_2 = compute_orientation_points(line_1.p0, line_1.p1, line_0.p0)
    orient_3 = compute_orientation_points(line_1.p0, line_1.p1, line_0.p1)

    # If the lines intersect, the triangle above the interseaction has
    # a reverse orientation as the triangle below, for both pairs of triagles.
    # (if only for one pair, then they do not need to intersect)
    if (orient_0 != orient_1) and (orient_2 != orient_3):
        return True

    # Now they may still intersect if they are both collinear and overlap.
    elif (orient_0 == Orientation.COLLINEAR) \
            and (orient_1 == Orientation.COLLINEAR) \
            and (orient_2 == Orientation.COLLINEAR) \
            and (orient_3 == Orientation.COLLINEAR):
        return do_collinear_lines_intersect(line_0, line_1)
    
    else:
        return False

def compute_orientation_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) 
        -> Orientation:
    """
    Returns whether three points are aligned in clockwise order.

    Returns:
        * Orientation: whether p0, p1, p2 are aligned clockwise,
            counterclockwise or collinear (i.e. aligned in a straight line).
    """
    # Theory:
    # https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/%c2%a0/
    slope_p0_p1 = compute_slope_two_points(p0, p1)
    slope_p1_p2 = compute_slope_two_points(p1, p2)

    if math.isclose(slope_p0_p1 == slope_p1_p2):
        return Orientation.COLLINEAR
    elif (slope_p0_p1 > slope_p1_p2):
        # A right turn is made from slope_p0_p1 to slope_p1_p2
        return Orientation.CLOCKWISE
    else:
        return Orientation.COUNTERCLOCKWISE

def compute_slope_two_points(p0: np.ndarray, p1: np.ndarray) -> Number:
    # Slope = Dy / Dx
    return (p1[1]-p0[1]) / (p1[0] - p0[0])

def do_collinear_lines_intersect(line_0: Line, line_1: Line) -> bool:
    # Check if x-projections and y-projections intersect.
    if do_intervals_intersect(line_0.p0[0], line_0.p1[0], line_1.p0[0], 
            line_1.p1[0]) \
        and do_intervals_intersect(line_0.p0[1], line_0.p1[1], line_1.p0[1], 
                line_1.p1[1]):
            return True
    else:
        return False

def do_intervals_intersect(a0:Number, a1:Number, b0:Number, b1:Number)-> bool:
    """
    Returns whether two 1-dimensional intervals [a0, a1] and [b0, b1]
    intersect.
    """
    return ((a0 <= b0) and (a1 >= b0)) or ((a0 <= b1) and (a1 >= b1))

def dist_point_segment_line(point: np.ndarray, line: Line):
    """
    Get the (shortest) distance of a Point to a Line,
    treating the Line as a finite line segment.
    """
    # Derivation: http://geomalgorithms.com/a02-_lines.html 
    v = line.p1 - line.p0
    w = point - line.p0

    # Check if point beyond p0-end of line
    c = np.dot(v, w)
    if (c <= 0):
        return euclidean_dist(point, line.p0)
    # Check if point beyond p1-end of line
    elif (np.dot(v, v) <= c):
        return euclidean_dist(point, line.p1)
    # Then return the distance to the (inf) line itself.
    return dist_point_inf_line(point, line)

def euclidean_dist(p0, p1):
    """
    Returns Euclidean distance between two points.
    """
    return LA.norm(p0 - p1)

def dist_point_inf_line(point, line):
    """
    Get the distance of a Point to a Line,
    treating the Line as an infinite line.
    """
    # Proof: https://brilliant.org/wiki/dot-product-distance-between-point-and-a-line/
    a, b, c = get_line_implicit_coefs(line)
    distance = abs(a * point[0] + b * point[1] + c ) \
                / math.sqrt(a**2 + b**2)
    return distance

def is_ball_in_rect(ball: Ball, size: Size) -> bool:
    """
    Returns whether the given ball is located in the rectangle from
    (0, 0) to (size.x, size.y) (and not touching any border of the rectangle).
    """
    # Check left and top edge.
    if (ball.pos[0] - ball.rad <= 0) or (ball.pos[1] - ball.rad <= 0):
        return False
    # Check right and bottom edge.
    if (ball.pos[0] + ball.rad >= size.x) or (ball.pos[1] + ball.rad >= size.y):
        return False
    else:
        return True

def check_validity_point(point: np.ndarray):
    if not isinstance(point, np.ndarray):
        raise ValueError("Invalid input point, np.ndarray expected.")
    if (len(point) != 2):
        raise ValueError("Invalid input point, length 2 expected.")
