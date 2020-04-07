"""
Model providing the maze-simulation of the environment.

Author: Lulof Pirée
"""
import math
import numpy as np
from numpy import linalg as LA

def dot(p1, p2):
    """
    Returns dot product of two points.
    """
    return p1.x*p2.x + p1.y*p2.y

class Size():
    """
    Record type for width & height of rectangular coordinate planes.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Point():
    """
    Record type for a point in a Cartesian plane.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class Line():
    """
    Record type for a straight line in Cartesian coordinate space.
    """

    def __init__(self, x1, y1, x2, y2):
        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x2, y2])
        #self.length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # Coefficients of equation ax + by + c = 0 for this line.
        

    def __str__(self):
        return f"Line ({self.p1.x}, {self.p1.y}) -> ({self.p2.x}, {self.p2.y})"

class Ball():
    """
    Record type for a moving ball, has a position, radius,
    velocity (in 2 dimensions) and acceleration (in 2 dimensions).
    """
    def __init__(self, x, y, rad, x_vel=0, y_vel=0, x_acc=0, y_acc=0):
        self.p = np.array([x, y])
        self.rad = rad
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.x_acc = x_acc
        self.y_acc = y_acc

class MazeLayout():

    def __init__(self, lines=set()):
        assert(isinstance(lines, set))
        self.lines = lines

    def get_ball_hits_wall(self, ball):
        assert(isinstance(ball, Ball))

        for line in self.lines:
            if (self.__collides(ball, line)):
                return True
        return False

    def __collides(self, ball, line):
        """
        Checks if a given ball collides with a given line.
        Note that just hitting without overlapping counts as hitting.
        Also note that lines have a finite length here.
        """
        distance = dist_point_segment_line(ball.p, line)
        
        if (distance <= ball.rad):
            return True
        else:
            return False


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
    x1 = line.p1[0]
    y1 = line.p1[1]
    x2 = line.p2[0]
    y2 = line.p2[1]
    
    a = y2 - y1
    b = x1 - x2
    c = y2*(x2 - x1) + x2*(y1 - y2)

    return a, b, c

def dist_point_segment_line(point, line):
    """
    Get the distance of a Point to a Line,
    treating the Line as a finite line segment.
    """
    # Derivation: http://geomalgorithms.com/a02-_lines.html 
    v = line.p2 - line.p1
    w = point - line.p1

    # Check if point beyond p1-end of line
    c = np.dot(v, w)
    if (c <= 0):
        return euclidean_dist(point, line.p1)
    # Check if point beyond p2-end of line
    elif (np.dot(v, v) <= c):
        return euclidean_dist(point, line.p2)
    # Then return the distance to the (inf) line itself.
    return dist_point_inf_line(point, line)

def euclidean_dist(p1, p2):
    """
    Returns Euclidean distance between two points.
    """
    return LA.norm(p1 - p2)

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
