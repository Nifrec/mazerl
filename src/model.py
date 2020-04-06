"""
Model providing the maze-simulation of the environment.

Author: Lulof Pirée
"""
import math

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

class Line():
    """
    Record type for a straight line in Cartesian coordinate space.
    """

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        #self.length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # Coefficients of equation ax + by + c = 0 for this line.
        # Proof:
        # [x2]   [x1-x2]          [x]
        # [y2] + [y1-y2]*lambda = [y]
        # 
        # --> lambda = (x - x2)/(x1-x2)
        # --> y = y2 + (y1-y2)*lambda = y2 + (y1-y2)(x - x2)/(x1-x2)
        # --> y(x1 - x2) + x(y2 - y1) + y2(x2 - x1) +x2(y1 - y2) = 0
        self.a = x1 - x2
        self.b = y2 - y1
        self.c = y2*(x2 - x1) + x2*(y1 - y2)

    def __str__(self):
        return f"Line ({self.x1}, {self.y1}) -> ({self.x2}, {self.y2})"

class Ball():
    """
    Record type for a moving ball, has a position, radius,
    velocity (in 2 dimensions) and acceleration (in 2 dimensions).
    """
    def __init__(self, x, y, rad, x_vel=0, y_vel=0, x_acc=0, y_acc=0):
        self.x = x
        self.y = y
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
            pass

    def __collides(self, ball, line):
        pass
