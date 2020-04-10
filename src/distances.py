"""
Auxiliary functions for the model that provides
the maze-simulation of the environment.
These auxiliary functions deal with distances in
2D Cartesian space.

Author: Lulof Pirée
"""
# Library imports:
import math
import enum
import numpy as np
from numpy import linalg as LA
from numbers import Number
# Local imports:
from record_types import Size, Line, Ball

class Orientation(enum.Enum):
    CLOCKWISE = 1
    COUNTERCLOCKWISE = 2
    COLLINEAR = 3


def __get_line_implicit_coefs(line):
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

def dist_line_segment_line_segment(line_0: Line,
        line_1: Line) -> Number:
    """
    Computes the shortest distance between two line segments.
    """
    # 2 cases: 
    # Case 1: the lines intersect -> dist = 0
    if (__do_lines_intersect(line_0, line_1)):
        return 0
    else:
        # Case 2: Return minimum distance of any eindpoint to the other line.
        d0 = dist_point_line_segment(line_0.p0, line_1)
        d1 = dist_point_line_segment(line_0.p1, line_1)
        d2 = dist_point_line_segment(line_1.p0, line_0)
        d3 = dist_point_line_segment(line_1.p1, line_0)
        return min(d0, d1, d2, d3)


def __do_lines_intersect(line_0: Line, line_1: Line) -> bool:
    """
    Returns whether two finite line segments intersect.
    """
    # Theory:
    # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    
    # Compute orientations of all triangles created by taking one line
    # and one endpoint of the other line.
    # For lines AB and XY, that are O(A, B, X), O(A, B, Y),
    # O(X, Y, A), O(X, Y, B) (where O is the orientation)
    orient_0 = __compute_orientation_points(line_0.p0, line_0.p1, line_1.p0)
    orient_1 = __compute_orientation_points(line_0.p0, line_0.p1, line_1.p1)
    orient_2 = __compute_orientation_points(line_1.p0, line_1.p1, line_0.p0)
    orient_3 = __compute_orientation_points(line_1.p0, line_1.p1, line_0.p1)

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
        return __do_collinear_lines_intersect(line_0, line_1)
    
    else:
        return False

def __compute_orientation_points(p0: np.ndarray, p1: np.ndarray,
        p2: np.ndarray) -> Orientation:
    """
    Returns whether three points are aligned in clockwise order.

    Returns:
        * Orientation: whether p0, p1, p2 are aligned clockwise,
            counterclockwise or collinear (i.e. aligned in a straight line).
    """
    # Theory:
    # https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/%c2%a0/
    slope_p0_p1 = __compute_slope_two_points(p0, p1)
    slope_p1_p2 = __compute_slope_two_points(p1, p2)

    if math.isclose(slope_p0_p1 == slope_p1_p2):
        return Orientation.COLLINEAR
    elif (slope_p0_p1 > slope_p1_p2):
        # A right turn is made from slope_p0_p1 to slope_p1_p2
        return Orientation.CLOCKWISE
    else:
        return Orientation.COUNTERCLOCKWISE

def __compute_slope_two_points(p0: np.ndarray, p1: np.ndarray) -> Number:
    # Slope = Dy / Dx
    return (p1[1]-p0[1]) / (p1[0] - p0[0])

def __do_collinear_lines_intersect(line_0: Line, 
        line_1: Line) -> bool:
    # Check if x-projections and y-projections intersect.
    if __do_intervals_intersect(line_0.p0[0], line_0.p1[0], line_1.p0[0], 
            line_1.p1[0]) \
        and __do_intervals_intersect(line_0.p0[1], line_0.p1[1], line_1.p0[1], 
                line_1.p1[1]):
            return True
    else:
        return False

def __do_intervals_intersect(a0:Number, a1:Number, b0:Number, b1:Number)-> bool:
    """
    Returns whether two 1-dimensional intervals [a0, a1] and [b0, b1]
    intersect.
    """
    return ((a0 <= b0) and (a1 >= b0)) or ((a0 <= b1) and (a1 >= b1))

def dist_point_line_segment(point: np.ndarray, line: Line):
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
    return __dist_point_inf_line(point, line)

def euclidean_dist(p0, p1):
    """
    Returns Euclidean distance between two points.
    """
    return LA.norm(p0 - p1)

def __dist_point_inf_line(point, line):
    """
    Get the distance of a Point to a Line,
    treating the Line as an infinite line.
    """
    # Proof: https://brilliant.org/wiki/dot-product-distance-between-point-and-a-line/
    a, b, c = __get_line_implicit_coefs(line)
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